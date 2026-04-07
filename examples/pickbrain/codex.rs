use anyhow::Result;
use regex::Regex;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

use uuid::Uuid;

use witchcraft::DB;

const MIN_CHUNK_CODEPOINTS: usize = 5;
const MAX_CHUNK_CODEPOINTS: usize = 4000;

const CODEX_NAMESPACE: Uuid = Uuid::from_bytes([
    0xb4, 0xe8, 0xd9, 0xe2, 0x7f, 0x3c, 0x5b, 0xa2, 0xc6, 0xe1, 0x90, 0x2f, 0x4d, 0x8b, 0xac,
    0x3e,
]);

#[derive(Deserialize)]
struct Entry {
    timestamp: Option<String>,
    #[serde(rename = "type")]
    entry_type: String,
    #[serde(default)]
    payload: serde_json::Value,
}

struct Chunk {
    role: String,
    text: String,
    timestamp: String,
    ts_ms: i64,
}

fn codepoint_len(s: &str) -> usize {
    s.chars().count()
}

fn sanitize(text: &str) -> String {
    let s = strip_xml(text);
    let s = strip_code(&s);
    compact(&s)
}

fn strip_xml(text: &str) -> String {
    let re = Regex::new(r"<[^>]+>").unwrap();
    re.replace_all(text, "").to_string()
}

fn strip_code(text: &str) -> String {
    let re = Regex::new(r"```[\s\S]*?```").unwrap();
    let s = re.replace_all(text, " ").to_string();
    let re = Regex::new(r"```[\s\S]*$").unwrap();
    let s = re.replace_all(&s, " ").to_string();
    let re = Regex::new(r"`[^`]*`").unwrap();
    re.replace_all(&s, " ").to_string()
}

fn compact(text: &str) -> String {
    let re = Regex::new(r"\s{2,}").unwrap();
    re.replace_all(text, " ").trim().to_string()
}

fn extract_user_text(payload: &serde_json::Value) -> Option<String> {
    if payload.get("type")?.as_str()? != "message" {
        return None;
    }
    if payload.get("role")?.as_str()? != "user" {
        return None;
    }
    let content = payload.get("content")?.as_array()?;
    let texts: Vec<&str> = content
        .iter()
        .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("input_text"))
        .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
        .collect();
    if texts.is_empty() {
        None
    } else {
        Some(texts.join("\n"))
    }
}

fn extract_reasoning_text(payload: &serde_json::Value) -> Option<String> {
    if payload.get("type")?.as_str()? != "agent_reasoning" {
        return None;
    }
    payload.get("text").and_then(|t| t.as_str()).map(|s| s.to_string())
}

fn parse_session_file(path: &Path) -> (Option<String>, Vec<Chunk>) {
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return (None, vec![]),
    };

    let mut cwd = None;
    let mut chunks = Vec::new();

    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let entry: Entry = match serde_json::from_str(line) {
            Ok(e) => e,
            Err(_) => continue,
        };

        if entry.entry_type == "session_meta" {
            if cwd.is_none() {
                cwd = entry.payload.get("cwd").and_then(|c| c.as_str()).map(|s| s.to_string());
            }
            continue;
        }

        let timestamp = match &entry.timestamp {
            Some(ts) if !ts.is_empty() => ts.clone(),
            _ => continue,
        };
        let ts_ms = chrono::DateTime::parse_from_rfc3339(&timestamp)
            .map(|dt| dt.timestamp_millis())
            .unwrap_or(0);
        if ts_ms <= 0 {
            continue;
        }

        let (role, raw_text) = if entry.entry_type == "response_item" {
            if let Some(text) = extract_user_text(&entry.payload) {
                ("user".to_string(), text)
            } else {
                continue;
            }
        } else if entry.entry_type == "event_msg" {
            if let Some(text) = extract_reasoning_text(&entry.payload) {
                ("assistant".to_string(), text)
            } else {
                continue;
            }
        } else {
            continue;
        };

        let text = sanitize(&raw_text);
        if text.is_empty() {
            continue;
        }
        let cp_len = codepoint_len(&text);
        if cp_len < MIN_CHUNK_CODEPOINTS || cp_len > MAX_CHUNK_CODEPOINTS {
            continue;
        }

        chunks.push(Chunk {
            role,
            text,
            timestamp,
            ts_ms,
        });
    }

    chunks.sort_by_key(|c| c.ts_ms);
    (cwd, chunks)
}

fn project_from_cwd(cwd: &str) -> String {
    Path::new(cwd)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| cwd.to_string())
}

fn session_id_from_filename(path: &Path) -> String {
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    // Filename: rollout-YYYY-MM-DDTHH-mm-ss-<UUID>
    // Extract the UUID part (last 36 chars of the stem)
    if stem.len() >= 36 {
        let uuid_part = &stem[stem.len() - 36..];
        if uuid_part.chars().filter(|&c| c == '-').count() == 4 {
            return uuid_part.to_string();
        }
    }
    stem.to_string()
}

fn ingest_session(db: &mut DB, path: &Path, mtime_ms: i64) -> Result<usize> {
    let (cwd, chunks) = parse_session_file(path);
    if chunks.is_empty() {
        return Ok(0);
    }

    let project_name = cwd.as_deref().map(project_from_cwd).unwrap_or_default();
    let session_id = session_id_from_filename(path);

    let session_title: String = chunks
        .iter()
        .find(|c| c.role == "user")
        .map(|c| c.text.chars().take(240).collect())
        .unwrap_or_default();

    // Split into interactions starting at each user message
    let mut interactions: Vec<&[Chunk]> = Vec::new();
    let mut start = 0;
    for (i, chunk) in chunks.iter().enumerate() {
        if chunk.role == "user" && i > start {
            interactions.push(&chunks[start..i]);
            start = i;
        }
    }
    interactions.push(&chunks[start..]);

    let mut count = 0;
    for (turn_idx, interaction) in interactions.iter().enumerate() {
        let header = format!("[codex:{project_name}] {session_title}\n");
        // One entry per turn so sub_idx maps 1:1 to conversation turns
        let mut all_parts = vec![header];
        let mut turns_meta: Vec<serde_json::Value> = Vec::new();

        for chunk in *interaction {
            let label = if chunk.role == "user" {
                "[User]"
            } else {
                "[Codex]"
            };
            all_parts.push(format!("{label} {}\n", chunk.text));
            turns_meta.push(serde_json::json!({
                "role": chunk.role,
                "timestamp": chunk.timestamp,
            }));
        }

        let lengths: Vec<usize> = all_parts.iter().map(|p| p.chars().count()).collect();
        let body = all_parts.join("");
        if body.trim().is_empty() {
            continue;
        }

        let uuid = Uuid::new_v5(
            &CODEX_NAMESPACE,
            format!("{session_id}:{turn_idx}").as_bytes(),
        );

        let metadata = serde_json::json!({
            "source": "codex",
            "project": project_name,
            "session_id": session_id,
            "turn": turn_idx,
            "path": path.to_string_lossy(),
            "cwd": cwd,
            "mtime_ms": mtime_ms,
            "turns": turns_meta,
        })
        .to_string();

        let date = iso8601_timestamp::Timestamp::parse(&interaction[0].timestamp);
        db.add_doc(&uuid, date, &metadata, &body, Some(lengths))?;
        count += 1;
    }

    Ok(count)
}

fn file_mtime_ms(path: &Path) -> Option<i64> {
    fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_millis() as i64)
}

use crate::watermark;

/// Walk ~/.codex/sessions/ recursively for .jsonl files
fn collect_session_files(base: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|ext| ext == "jsonl") {
                out.push(p);
            }
        }
    }
    walk(base, &mut files);
    files.sort();
    files
}

pub fn ingest_codex(db: &mut DB) -> Result<usize> {
    let home = std::env::var("HOME").unwrap_or_default();
    let sessions_dir = PathBuf::from(&home).join(".codex/sessions");

    if !sessions_dir.is_dir() {
        return Ok(0);
    }

    let wm_path = watermark::codex_path();
    let wm_ts = watermark::mtime_ms(&wm_path);
    let mut session_count = 0usize;

    for jsonl_path in collect_session_files(&sessions_dir) {
        if !watermark::file_newer_than(&jsonl_path, wm_ts) {
            continue;
        }
        let mtime_ms = file_mtime_ms(&jsonl_path).unwrap_or(0);
        println!("{}", jsonl_path.display());
        match ingest_session(db, &jsonl_path, mtime_ms) {
            Ok(n) => session_count += n,
            Err(e) => {
                log::warn!("failed to ingest codex {}: {e}", jsonl_path.display());
            }
        }
    }

    watermark::touch(&wm_path);
    Ok(session_count)
}
