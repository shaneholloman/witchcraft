use anyhow::Result;
use log::{Level, LevelFilter, Metadata, Record};
use std::env;
use std::io::Write;
use std::path::PathBuf;

mod claude_code;
mod codex;
mod slack;
mod watermark;

use witchcraft::{DB, Embedder};

struct SimpleLogger;
impl log::Log for SimpleLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Warn
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            eprintln!("[{}] {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

static LOGGER: SimpleLogger = SimpleLogger;

fn db_path() -> PathBuf {
    let home = env::var("HOME").unwrap_or_default();
    let dir = PathBuf::from(home).join(".pickbrain");
    std::fs::create_dir_all(&dir).ok();
    dir.join("pickbrain.db")
}

fn assets_path() -> PathBuf {
    PathBuf::from(env::var("WARP_ASSETS").unwrap_or_else(|_| "assets".into()))
}

/// Walk up the process tree to find the calling Claude Code session ID.
fn detect_active_session() -> Option<String> {
    let home = env::var("HOME").ok()?;
    let sessions_dir = PathBuf::from(&home).join(".claude/sessions");
    let mut pid = std::process::id() as i32;
    while pid > 1 {
        let session_file = sessions_dir.join(format!("{pid}.json"));
        if let Ok(data) = std::fs::read_to_string(&session_file) {
            let v: serde_json::Value = serde_json::from_str(&data).ok()?;
            return v["sessionId"].as_str().map(|s| s.to_string());
        }
        pid = get_ppid(pid)?;
    }
    None
}

#[cfg(target_os = "macos")]
fn get_ppid(pid: i32) -> Option<i32> {
    let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_bsdinfo>() as i32;
    let ret = unsafe {
        libc::proc_pidinfo(pid, libc::PROC_PIDTBSDINFO, 0,
            &mut info as *mut _ as *mut libc::c_void, size)
    };
    if ret == size {
        let ppid = info.pbi_ppid as i32;
        if ppid > 0 { Some(ppid) } else { None }
    } else {
        None
    }
}

#[cfg(target_os = "linux")]
fn get_ppid(pid: i32) -> Option<i32> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // Field 4 (0-indexed: 3) is ppid. Fields 1 is (comm) which may contain spaces,
    // so skip past the closing paren first.
    let after_comm = stat.rfind(')')? + 2;
    let fields: Vec<&str> = stat[after_comm..].split_whitespace().collect();
    // fields[0] = state, fields[1] = ppid
    let ppid: i32 = fields.get(1)?.parse().ok()?;
    if ppid > 0 { Some(ppid) } else { None }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn get_ppid(_pid: i32) -> Option<i32> {
    None
}

fn ingest(db_name: &PathBuf, skip_session: Option<&str>, stale_ms: i64, types: &[String]) -> Result<bool> {
    let mut db = DB::new(db_name.clone()).unwrap();

    let want = |src: &str| types.is_empty() || types.iter().any(|t| t == src);

    if db.was_recreated() {
        watermark::remove(&watermark::claude_path());
        watermark::remove(&watermark::codex_path());
        slack::remove_watermark();
    }

    let (sessions, memories, authored, configs) = if want("claude") {
        let claude_skip = skip_session
            .filter(|_| watermark::is_fresh(&watermark::claude_path(), stale_ms));
        claude_code::ingest_claude_code(&mut db, claude_skip)?
    } else {
        (0, 0, 0, 0)
    };

    let codex_sessions = if want("codex") {
        codex::ingest_codex(&mut db)?
    } else {
        0
    };

    let slack_conversations = if want("slack") {
        slack::ingest_slack(&mut db)?
    } else {
        0
    };

    let total = sessions + memories + authored + configs + codex_sessions + slack_conversations;
    if total == 0 {
        eprintln!("No new sessions to ingest.");
        return Ok(false);
    }
    eprintln!(
        "ingested {sessions} claude sessions, {codex_sessions} codex sessions, {slack_conversations} slack conversations, {memories} memory files, {authored} authored files, {configs} config files"
    );
    Ok(true)
}

fn embed_and_index(db: &DB, embedder: &Embedder, device: &candle_core::Device) -> Result<()> {
    let embedded = witchcraft::embed_chunks(db, embedder, None)?;
    if embedded > 0 {
        witchcraft::index_chunks(db, device)?;
    }
    Ok(())
}

// --- Search result data ---

struct TurnMeta {
    role: String,
    timestamp: String,
    byte_offset: u64,
    byte_len: u64,
}

struct SearchResult {
    timestamp: String,
    project: String,
    session_id: String,
    turn: u64,
    path: String,
    cwd: String,
    source: String,
    conv_key: String,
    bodies: Vec<String>,
    match_idx: usize,
    turns: Vec<TurnMeta>,
}

// A turn from the original JSONL session file
struct SessionTurn {
    role: String,
    text: String,
    timestamp: String,
}

fn read_jsonl_line(path: &str, offset: u64, len: u64) -> Option<String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).ok()?;
    f.seek(SeekFrom::Start(offset)).ok()?;
    let mut buf = vec![0u8; len as usize];
    f.read_exact(&mut buf).ok()?;
    String::from_utf8(buf).ok()
}

fn read_turn_at(path: &str, source: &str, tm: &TurnMeta) -> Option<SessionTurn> {
    let line = read_jsonl_line(path, tm.byte_offset, tm.byte_len)?;
    let v: serde_json::Value = serde_json::from_str(&line).ok()?;

    let text = if source == "codex" {
        let payload = v.get("payload")?;
        let ptype = payload.get("type")?.as_str()?;
        if ptype == "message" && payload.get("role")?.as_str()? == "user" {
            let content = payload.get("content")?.as_array()?;
            let texts: Vec<&str> = content
                .iter()
                .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("input_text"))
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect();
            texts.join("\n")
        } else if ptype == "agent_reasoning" {
            payload.get("text")?.as_str()?.to_string()
        } else {
            return None;
        }
    } else {
        // Claude Code
        let msg = v.get("message")?;
        match msg.get("content")? {
            c if c.is_string() => c.as_str()?.to_string(),
            c if c.is_array() => {
                let blocks = c.as_array()?;
                blocks
                    .iter()
                    .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
                    .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            _ => return None,
        }
    };

    Some(SessionTurn {
        role: tm.role.clone(),
        text,
        timestamp: tm.timestamp.clone(),
    })
}


/// Parse a duration string like "24h", "7d", "2w" into milliseconds.
fn parse_since(s: &str) -> Option<i64> {
    let s = s.trim();
    let (num_str, unit) = s.split_at(s.len().saturating_sub(1));
    let n: i64 = num_str.parse().ok()?;
    let ms_per_unit = match unit {
        "h" => 3_600_000,
        "d" => 86_400_000,
        "w" => 604_800_000,
        _ => return None,
    };
    Some(n * ms_per_unit)
}

fn run_search(
    db_name: &PathBuf,
    assets: &PathBuf,
    q: &str,
    session: Option<&str>,
    exclude: &[String],
    since_ms: Option<i64>,
    types: &[String],
    num_results: usize,
    dm_only: bool,
    no_dm: bool,
    unread_only: bool,
) -> Result<(Vec<SearchResult>, u128)> {
    use witchcraft::types::*;
    let device = witchcraft::make_device();
    let embedder = witchcraft::Embedder::new(&device, assets)?;

    let mut cache = witchcraft::EmbeddingsCache::new(1);
    let db = DB::new_reader(db_name.clone()).unwrap();

    let session_filter = session.map(|id| {
        let name = id.strip_prefix('#').unwrap_or(id);
        // For thread references like "thr:1234.5678", extract just the TS part.
        let thread_ts = name.strip_prefix("thr:").unwrap_or(name);
        let thread_like = format!("%-{thread_ts}");
        // Match session_id (Claude/Codex), channel_id, channel_name, or conv_key thread TS (Slack)
        SqlStatementInternal {
            statement_type: SqlStatementType::Group,
            condition: None,
            logic: Some(SqlLogic::Or),
            statements: Some(vec![
                SqlStatementInternal {
                    statement_type: SqlStatementType::Condition,
                    condition: Some(SqlConditionInternal {
                        key: "$.session_id".to_string(),
                        operator: SqlOperator::Equals,
                        value: Some(SqlValue::String(id.to_string())),
                    }),
                    logic: None,
                    statements: None,
                },
                SqlStatementInternal {
                    statement_type: SqlStatementType::Condition,
                    condition: Some(SqlConditionInternal {
                        key: "$.channel_id".to_string(),
                        operator: SqlOperator::Equals,
                        value: Some(SqlValue::String(id.to_string())),
                    }),
                    logic: None,
                    statements: None,
                },
                SqlStatementInternal {
                    statement_type: SqlStatementType::Condition,
                    condition: Some(SqlConditionInternal {
                        key: "$.channel_name".to_string(),
                        operator: SqlOperator::Equals,
                        value: Some(SqlValue::String(name.to_string())),
                    }),
                    logic: None,
                    statements: None,
                },
                SqlStatementInternal {
                    statement_type: SqlStatementType::Condition,
                    condition: Some(SqlConditionInternal {
                        key: "$.conv_key".to_string(),
                        operator: SqlOperator::Like,
                        value: Some(SqlValue::String(thread_like)),
                    }),
                    logic: None,
                    statements: None,
                },
            ]),
        }
    });

    let exclude_filter = if exclude.is_empty() {
        None
    } else {
        let stmts: Vec<SqlStatementInternal> = exclude
            .iter()
            .map(|id| SqlStatementInternal {
                statement_type: SqlStatementType::Condition,
                condition: Some(SqlConditionInternal {
                    key: "$.session_id".to_string(),
                    operator: SqlOperator::NotEquals,
                    value: Some(SqlValue::String(id.clone())),
                }),
                logic: None,
                statements: None,
            })
            .collect();
        Some(SqlStatementInternal {
            statement_type: SqlStatementType::Group,
            condition: None,
            logic: Some(SqlLogic::And),
            statements: Some(stmts),
        })
    };

    let since_filter = since_ms.map(|ms| {
        let cutoff_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
            - ms / 1000;
        let cutoff_dt = chrono::DateTime::from_timestamp(cutoff_secs, 0).unwrap();
        let cutoff_iso = cutoff_dt.to_rfc3339();
        SqlStatementInternal {
            statement_type: SqlStatementType::Condition,
            condition: Some(SqlConditionInternal {
                key: "date".to_string(),
                operator: SqlOperator::GreaterThanOrEquals,
                value: Some(SqlValue::String(cutoff_iso)),
            }),
            logic: None,
            statements: None,
        }
    });

    let type_filter = if types.is_empty() {
        None
    } else if types.len() == 1 {
        Some(SqlStatementInternal {
            statement_type: SqlStatementType::Condition,
            condition: Some(SqlConditionInternal {
                key: "$.source".to_string(),
                operator: SqlOperator::Equals,
                value: Some(SqlValue::String(types[0].clone())),
            }),
            logic: None,
            statements: None,
        })
    } else {
        let stmts: Vec<SqlStatementInternal> = types
            .iter()
            .map(|t| SqlStatementInternal {
                statement_type: SqlStatementType::Condition,
                condition: Some(SqlConditionInternal {
                    key: "$.source".to_string(),
                    operator: SqlOperator::Equals,
                    value: Some(SqlValue::String(t.clone())),
                }),
                logic: None,
                statements: None,
            })
            .collect();
        Some(SqlStatementInternal {
            statement_type: SqlStatementType::Group,
            condition: None,
            logic: Some(SqlLogic::Or),
            statements: Some(stmts),
        })
    };

    let dm_filter = if dm_only {
        Some(SqlStatementInternal {
            statement_type: SqlStatementType::Condition,
            condition: Some(SqlConditionInternal {
                key: "$.is_dm".to_string(),
                operator: SqlOperator::Equals,
                value: Some(SqlValue::Number(1.0)),
            }),
            logic: None,
            statements: None,
        })
    } else if no_dm {
        Some(SqlStatementInternal {
            statement_type: SqlStatementType::Condition,
            condition: Some(SqlConditionInternal {
                key: "$.is_dm".to_string(),
                operator: SqlOperator::Equals,
                value: Some(SqlValue::Number(0.0)),
            }),
            logic: None,
            statements: None,
        })
    } else {
        None
    };

    let unread_filter = if unread_only {
        Some(SqlStatementInternal {
            statement_type: SqlStatementType::Condition,
            condition: Some(SqlConditionInternal {
                key: "$.has_unread".to_string(),
                operator: SqlOperator::Equals,
                value: Some(SqlValue::Number(1.0)),
            }),
            logic: None,
            statements: None,
        })
    } else {
        None
    };

    let filters: Vec<SqlStatementInternal> = [session_filter, exclude_filter, since_filter, type_filter, dm_filter, unread_filter]
        .into_iter()
        .flatten()
        .collect();

    let sql_filter = match filters.len() {
        0 => None,
        1 => Some(filters.into_iter().next().unwrap()),
        _ => Some(SqlStatementInternal {
            statement_type: SqlStatementType::Group,
            condition: None,
            logic: Some(SqlLogic::And),
            statements: Some(filters),
        }),
    };
    let effective_limit = if num_results == 0 { usize::MAX } else { num_results };
    let now = std::time::Instant::now();
    let results = if q.is_empty() {
        // Filter-only mode: return most recent matching documents
        use witchcraft::sql_generator::build_filter_sql_and_params;
        let (filter_sql, params) = build_filter_sql_and_params(sql_filter.as_ref())?;
        let where_clause = if filter_sql.is_empty() { String::new() } else { format!("WHERE {filter_sql}") };
        let limit_clause = if num_results == 0 { String::new() } else { " LIMIT ?".to_string() };
        let sql = format!(
            "SELECT metadata, body, lens, date FROM document {where_clause} ORDER BY date DESC{limit_clause}",
        );
        let mut stmt = db.query(&sql)?;
        let mut param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| &**p as &dyn rusqlite::ToSql).collect();
        let limit = num_results as i64;
        if num_results > 0 {
            param_refs.push(&limit);
        }
        let rows: Vec<(f32, String, Vec<String>, u32, String)> = stmt
            .query_map(param_refs.as_slice(), |row| {
                let metadata: String = row.get(0)?;
                let body: String = row.get(1)?;
                let lens_str: String = row.get(2)?;
                let date: String = row.get(3)?;
                let lens: Vec<usize> = lens_str.split(',').map(|x| x.parse::<usize>().unwrap_or(0)).collect();
                let bodies: Vec<String> = witchcraft::split_by_codepoints(&body, &lens)
                    .into_iter().map(|s| s.to_string()).collect();
                Ok((0.0f32, metadata, bodies, 0u32, date))
            })?
            .filter_map(|r| r.ok())
            .collect();
        rows
    } else {
        witchcraft::search(
            &db,
            &embedder,
            &mut cache,
            q,
            0.5,
            effective_limit,
            true,
            sql_filter.as_ref(),
        )?
    };
    let search_ms = now.elapsed().as_millis();

    let out: Vec<SearchResult> = results
        .into_iter()
        .map(|(_score, metadata, bodies, sub_idx, date)| {
            let meta: serde_json::Value = serde_json::from_str(&metadata).unwrap_or_default();
            let idx = (sub_idx as usize).min(bodies.len().saturating_sub(1));
            let turns_arr: Vec<TurnMeta> = meta["turns"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .map(|v| TurnMeta {
                            role: v["role"].as_str().unwrap_or("").to_string(),
                            timestamp: v["timestamp"].as_str().unwrap_or("").to_string(),
                            byte_offset: v["off"].as_u64().unwrap_or(0),
                            byte_len: v["len"].as_u64().unwrap_or(0),
                        })
                        .collect()
                })
                .unwrap_or_default();
            SearchResult {
                timestamp: format_date(&date),
                project: meta["project"].as_str().unwrap_or("").to_string(),
                session_id: meta["session_id"].as_str().unwrap_or("").to_string(),
                turn: meta["turn"].as_u64().unwrap_or(0),
                path: meta["path"].as_str().unwrap_or("").to_string(),
                cwd: meta["cwd"].as_str().unwrap_or("").to_string(),
                source: meta["source"].as_str().unwrap_or("claude").to_string(),
                conv_key: meta["conv_key"].as_str().unwrap_or("").to_string(),
                bodies,
                match_idx: idx,
                turns: turns_arr,
            }
        })
        .collect();
    Ok((out, search_ms))
}

// --- TUI ---

enum View {
    List,
    Detail(usize),
}

fn search_tui(
    db_name: &PathBuf,
    assets: &PathBuf,
    q: &str,
    session: Option<&str>,
    exclude: &[String],
    since_ms: Option<i64>,
    types: &[String],
    num_results: usize,
    dm_only: bool,
    no_dm: bool,
    unread_only: bool,
) -> Result<Option<(String, String, String)>> {
    let (results, search_ms) = run_search(db_name, assets, q, session, exclude, since_ms, types, num_results, dm_only, no_dm, unread_only)?;
    if results.is_empty() {
        eprintln!("no results");
        return Ok(None);
    }

    use crossterm::event::{self, Event, KeyCode, KeyModifiers};
    use crossterm::terminal::{
        disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
    };
    use ratatui::backend::CrosstermBackend;
    use ratatui::layout::{Constraint, Direction, Layout};
    use ratatui::style::{Color, Modifier, Style};
    use ratatui::text::{Line, Span};
    use ratatui::widgets::{ListState, Paragraph, Wrap};
    use ratatui::Terminal;

    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    crossterm::execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut view = View::List;
    let mut selected: usize = 0;
    let mut list_state = ListState::default();
    list_state.select(Some(0));
    let mut scroll_offset: usize = 0;
    let mut resume_session: Option<(String, String, String)> = None;
    let mut confirm_resume: Option<(String, String, String, String)> = None;
    struct DetailState {
        result_idx: usize,
        turns: Vec<SessionTurn>,
        highlight: usize,
    }
    let mut detail_cache: Option<DetailState> = None;

    loop {
        terminal.draw(|f| {
            let area = f.area();
            let show_footer = confirm_resume.is_some() && matches!(view, View::Detail(_));
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(if show_footer {
                    vec![Constraint::Length(2), Constraint::Min(0), Constraint::Length(1)]
                } else {
                    vec![Constraint::Length(2), Constraint::Min(0), Constraint::Length(0)]
                })
                .split(area);

            // Header
            let header = Paragraph::new(Line::from(vec![
                Span::styled(
                    format!("[[ {q} ]]"),
                    Style::default()
                        .fg(Color::Rgb(0, 255, 0))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    slack::cached_self_user()
                        .map(|(_id, name)| format!("  (you: {name})"))
                        .unwrap_or_default(),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    format!("  {search_ms} ms  "),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    match view {
                        View::List => "↑↓ navigate  ⏎ open  q quit",
                        View::Detail(idx) if !results[idx].session_id.is_empty() => {
                            "↑↓ scroll  r resume session  esc back  q quit"
                        }
                        View::Detail(_) => "↑↓ scroll  esc back  q quit",
                    },
                    Style::default().fg(Color::DarkGray),
                ),
            ]));
            f.render_widget(header, chunks[0]);

            // Footer: resume confirmation
            if show_footer {
                let cwd = confirm_resume.as_ref()
                    .map(|(_, _, c, _)| c.as_str())
                    .unwrap_or("?");
                let sid = confirm_resume.as_ref()
                    .map(|(s, _, _, _)| s.as_str())
                    .unwrap_or("?");
                let src = confirm_resume.as_ref()
                    .map(|(_, _, _, s)| s.as_str())
                    .unwrap_or("claude");
                let footer = Paragraph::new(Line::from(vec![
                    Span::styled(
                        format!(" Exit pickbrain and resume {src} session {sid} in {cwd}? "),
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        "(Y/n)",
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                ]));
                f.render_widget(footer, chunks[2]);
            }

            match view {
                View::List => {
                    let width = chunks[1].width as usize;
                    let items: Vec<ratatui::widgets::ListItem> = results
                        .iter()
                        .map(|r| {
                            let preview_idx = if r.match_idx == 0 && r.bodies.len() > 1 {
                                1
                            } else {
                                r.match_idx
                            };
                            let raw_preview = first_line(&r.bodies[preview_idx]);
                            let preview = strip_body_prefix(&raw_preview);
                            let matched_tm = if r.match_idx > 0 {
                                r.turns.get(r.match_idx - 1)
                            } else {
                                r.turns.first()
                            };
                            let ts = matched_tm
                                .filter(|tm| !tm.timestamp.is_empty())
                                .map(|tm| format_date(&tm.timestamp))
                                .unwrap_or_else(|| r.timestamp.clone());
                            let mut meta_spans = vec![
                                Span::styled(
                                    format!("{ts} "),
                                    Style::default().fg(Color::Green),
                                ),
                                Span::styled(&r.project, Style::default().fg(Color::Cyan)),
                            ];
                            if r.path.ends_with(".md") {
                                meta_spans.push(Span::styled(
                                    format!("  {}", r.path),
                                    Style::default().fg(Color::Yellow),
                                ));
                            }
                            if r.source == "slack" {
                                let tag = slack_conv_tag(&r.conv_key);
                                meta_spans.push(Span::styled(
                                    format!("  slack {tag}"),
                                    Style::default().fg(Color::Magenta),
                                ));
                            } else if !r.session_id.is_empty() {
                                let source_label = if r.source == "codex" {
                                    "codex"
                                } else {
                                    "claude"
                                };
                                let short_sid = if r.session_id.len() > 8 {
                                    &r.session_id[..8]
                                } else {
                                    &r.session_id
                                };
                                meta_spans.push(Span::styled(
                                    format!("  {source_label} {short_sid}  turn {}", r.turn),
                                    Style::default().fg(Color::Magenta),
                                ));
                            }
                            let match_role = matched_tm.map(|tm| tm.role.as_str()).unwrap_or("");
                            let role_prefix = if r.source == "slack" {
                                ""
                            } else if match_role == "user" {
                                "[User] "
                            } else if match_role == "assistant" {
                                if r.source == "codex" { "[Codex] " } else { "[Claude] " }
                            } else {
                                ""
                            };
                            ratatui::widgets::ListItem::new(vec![
                                Line::from(meta_spans),
                                Line::from(vec![
                                    Span::styled(
                                        format!("  {role_prefix}"),
                                        Style::default().fg(if match_role == "user" {
                                            Color::Rgb(0, 255, 0)
                                        } else {
                                            Color::Cyan
                                        }),
                                    ),
                                    Span::raw(truncate(&preview, width.saturating_sub(4 + role_prefix.len()))),
                                ]),
                                Line::from(""),
                            ])
                        })
                        .collect();

                    let list = ratatui::widgets::List::new(items).highlight_style(
                        Style::default()
                            .bg(Color::DarkGray)
                            .add_modifier(Modifier::BOLD),
                    );
                    f.render_stateful_widget(list, chunks[1], &mut list_state);
                }
                View::Detail(idx) => {
                    let r = &results[idx];
                    let mut lines: Vec<Line> = Vec::new();

                    // Session header
                    lines.push(Line::from(vec![
                        Span::styled(
                            format!("{} ", r.timestamp),
                            Style::default().fg(Color::Green),
                        ),
                        Span::styled(&r.project, Style::default().fg(Color::Cyan)),
                    ]));
                    if !r.session_id.is_empty() {
                        lines.push(Line::from(vec![
                            Span::styled(&r.session_id, Style::default().fg(Color::Magenta)),
                            Span::styled(
                                format!("  turn {}", r.turn),
                                Style::default().fg(Color::DarkGray),
                            ),
                        ]));
                    }
                    lines.push(Line::from(""));

                    // If we have a JSONL path and a session, show the real conversation
                    let dw = detail_cache
                        .as_ref()
                        .filter(|dw| dw.result_idx == idx);

                    if let Some(dw) = dw {
                        for (i, turn) in dw.turns.iter().enumerate() {
                            let is_highlight = i == dw.highlight;
                            let role_style = if turn.role == "user" {
                                Style::default()
                                    .fg(Color::Rgb(0, 255, 0))
                                    .add_modifier(Modifier::BOLD)
                            } else {
                                Style::default()
                                    .fg(Color::Cyan)
                                    .add_modifier(Modifier::BOLD)
                            };
                            lines.push(Line::from(vec![
                                Span::styled(
                                    if r.source == "slack" {
                                        ""
                                    } else if turn.role == "user" {
                                        "[User] "
                                    } else if r.source == "codex" {
                                        "[Codex] "
                                    } else {
                                        "[Claude] "
                                    },
                                    role_style,
                                ),
                                Span::styled(
                                    format_date(&turn.timestamp),
                                    Style::default().fg(Color::DarkGray),
                                ),
                            ]));
                            let text_style = if is_highlight {
                                Style::default().fg(Color::White)
                            } else {
                                Style::default().fg(Color::DarkGray)
                            };
                            for line in turn.text.lines() {
                                lines.push(Line::styled(format!("  {line}"), text_style));
                            }
                            lines.push(Line::from(""));
                        }
                    } else {
                        // Fallback: show indexed bodies (for .md files etc.)
                        for (i, chunk) in r.bodies.iter().enumerate() {
                            let style = if i == r.match_idx {
                                Style::default().add_modifier(Modifier::BOLD)
                            } else {
                                Style::default().fg(Color::DarkGray)
                            };
                            for line in chunk.lines().filter(|l| !l.is_empty()) {
                                lines.push(Line::styled(format!("  {line}"), style));
                            }
                            lines.push(Line::from(""));
                        }
                    }

                    let detail = Paragraph::new(lines)
                        .wrap(Wrap { trim: false })
                        .scroll((scroll_offset as u16, 0));
                    f.render_widget(detail, chunks[1]);
                }
            }
        })?;

        if let Event::Key(key) = event::read()? {
            match (&view, key.code, key.modifiers) {
                (_, KeyCode::Char('q') | KeyCode::Esc, _) if confirm_resume.is_some() => {
                    confirm_resume = None;
                }
                (View::Detail(_), KeyCode::Char('q') | KeyCode::Esc, _) => {
                    view = View::List;
                }
                (View::List, KeyCode::Char('q') | KeyCode::Esc, _) => break,
                (_, KeyCode::Char('c'), KeyModifiers::CONTROL) => break,
                (_, KeyCode::Char('z'), KeyModifiers::CONTROL) => {
                    disable_raw_mode()?;
                    crossterm::execute!(
                        std::io::stdout(),
                        LeaveAlternateScreen,
                        crossterm::cursor::Show
                    )?;
                    unsafe { libc::raise(libc::SIGTSTP); }
                    // When resumed (fg), re-enter TUI
                    enable_raw_mode()?;
                    crossterm::execute!(
                        std::io::stdout(),
                        EnterAlternateScreen,
                        crossterm::cursor::Hide
                    )?;
                    // Force full redraw
                    terminal.clear()?;
                }

                // List view
                (View::List, KeyCode::Down | KeyCode::Char('j'), _) => {
                    if selected + 1 < results.len() {
                        selected += 1;
                        list_state.select(Some(selected));
                    }
                }
                (View::List, KeyCode::Up | KeyCode::Char('k'), _) => {
                    if selected > 0 {
                        selected -= 1;
                        list_state.select(Some(selected));
                    }
                }
                (View::List, KeyCode::Enter, _) => {
                    let r = &results[selected];
                    if !r.session_id.is_empty() && !r.path.is_empty() && !r.turns.is_empty() {
                        let mi = if r.match_idx > 0 { r.match_idx - 1 } else { 0 };
                        let mut turns = Vec::new();
                        for tm in &r.turns {
                            if let Some(turn) = read_turn_at(&r.path, &r.source, tm) {
                                turns.push(turn);
                            }
                        }
                        // Position viewport at the highlight turn
                        let mut pre_lines: usize = if r.session_id.is_empty() { 2 } else { 3 };
                        for t in &turns[..mi.min(turns.len())] {
                            pre_lines += 2 + t.text.lines().count();
                        }
                        scroll_offset = pre_lines;
                        detail_cache = Some(DetailState {
                            result_idx: selected,
                            turns,
                            highlight: mi,
                        });
                    } else {
                        scroll_offset = 0;
                        detail_cache = None;
                    }
                    view = View::Detail(selected);
                }

                // Detail view: j/k line scroll
                (View::Detail(_), KeyCode::Down | KeyCode::Char('j'), _) => {
                    scroll_offset = scroll_offset.saturating_add(1);
                }
                (View::Detail(_), KeyCode::Up | KeyCode::Char('k'), _) => {
                    scroll_offset = scroll_offset.saturating_sub(1);
                }
                (View::Detail(idx), KeyCode::Char('r'), _) => {
                    let r = &results[*idx];
                    if !r.session_id.is_empty() {
                        let cwd = if !r.cwd.is_empty() {
                            r.cwd.clone()
                        } else {
                            read_cwd_from_jsonl(&r.path, &r.source)
                                .unwrap_or_else(|| "?".to_string())
                        };
                        confirm_resume = Some((r.session_id.clone(), r.path.clone(), cwd, r.source.clone()));
                    }
                }
                (View::Detail(_), KeyCode::Char('y') | KeyCode::Enter, _) => {
                    if let Some((ref sid, ref path, _, ref source)) = confirm_resume {
                        resume_session = Some((sid.clone(), path.clone(), source.clone()));
                        break;
                    }
                }
                (View::Detail(_), KeyCode::Char('n'), _) => {
                    confirm_resume = None;
                }
                _ => {}
            }
        }
    }

    disable_raw_mode()?;
    crossterm::execute!(std::io::stdout(), LeaveAlternateScreen)?;
    Ok(resume_session)
}

fn read_cwd_from_jsonl(path: &str, source: &str) -> Option<String> {
    let raw = std::fs::read_to_string(path).ok()?;
    for line in raw.lines() {
        let v: serde_json::Value = serde_json::from_str(line).ok()?;
        if source == "codex" {
            // Codex: cwd is in payload of session_meta entries
            if v.get("type").and_then(|t| t.as_str()) == Some("session_meta") {
                if let Some(cwd) = v.get("payload").and_then(|p| p.get("cwd")).and_then(|c| c.as_str()) {
                    return Some(cwd.to_string());
                }
            }
        } else {
            // Claude: cwd is a top-level field
            if let Some(cwd) = v.get("cwd").and_then(|c| c.as_str()) {
                return Some(cwd.to_string());
            }
        }
    }
    None
}

fn launch_resume(session_id: &str, jsonl_path: &str, source: &str) -> Result<()> {
    use std::os::unix::process::CommandExt;
    if let Some(cwd) = read_cwd_from_jsonl(jsonl_path, source) {
        let _ = std::env::set_current_dir(&cwd);
    }
    if source == "codex" {
        eprintln!("Resuming codex session {session_id}...");
        let err = std::process::Command::new("codex")
            .args(["resume", session_id])
            .exec();
        Err(err.into())
    } else {
        eprintln!("Resuming claude session {session_id}...");
        let err = std::process::Command::new("claude")
            .args(["--resume", session_id])
            .exec();
        Err(err.into())
    }
}


/// Extract a display tag from a Slack conv_key.
/// Thread keys (`thr:CHAN-TS`) → `thr:TS` (the thread timestamp, usable with --session).
/// Session keys (`sess:CHAN-TS`) → empty (channel + turn is sufficient).
fn slack_conv_tag(conv_key: &str) -> String {
    if let Some(rest) = conv_key.strip_prefix("thr:") {
        if let Some((_chan, ts)) = rest.split_once('-') {
            return format!("thr:{ts}");
        }
    }
    String::new()
}

fn strip_body_prefix(s: &str) -> &str {
    s.strip_prefix("[User] ")
        .or_else(|| s.strip_prefix("[Claude] "))
        .or_else(|| s.strip_prefix("[Codex] "))
        .unwrap_or(s)
}

fn first_line(text: &str) -> String {
    text.lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .to_string()
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let end = s.floor_char_boundary(max.saturating_sub(3));
        format!("{}...", &s[..end])
    }
}

// --- Plain text fallback (piped output) ---

fn search_plain(
    db_name: &PathBuf,
    assets: &PathBuf,
    q: &str,
    session: Option<&str>,
    exclude: &[String],
    since_ms: Option<i64>,
    types: &[String],
    num_results: usize,
    dm_only: bool,
    no_dm: bool,
    unread_only: bool,
) -> Result<()> {
    let (results, search_ms) = run_search(db_name, assets, q, session, exclude, since_ms, types, num_results, dm_only, no_dm, unread_only)?;

    let mut buf = Vec::new();
    let you = slack::cached_self_user()
        .map(|(_id, name)| format!("  (you: {name})"))
        .unwrap_or_default();
    writeln!(buf, "\n[[ {q} ]]{you}")?;
    writeln!(buf, "search completed in {search_ms} ms\n")?;
    for r in &results {
        writeln!(buf, "---")?;
        // match_idx 0 = header, turns[0] = first turn → turns[match_idx - 1]
        let matched_tm = if r.match_idx > 0 {
            r.turns.get(r.match_idx - 1)
        } else {
            r.turns.first()
        };
        let ts = matched_tm
            .filter(|tm| !tm.timestamp.is_empty())
            .map(|tm| format_date(&tm.timestamp))
            .unwrap_or_else(|| r.timestamp.clone());
        let source_label = if r.source == "slack" { "slack" } else if r.source == "codex" { "codex" } else { "claude" };
        let filename = if r.path.ends_with(".md") {
            format!("  {}", r.path)
        } else {
            String::new()
        };
        let session_info = if r.source == "slack" {
            let tag = slack_conv_tag(&r.conv_key);
            format!("  {source_label} {tag}")
        } else if !r.session_id.is_empty() {
            format!("  {source_label} {} turn {}", r.session_id, r.turn)
        } else {
            String::new()
        };
        writeln!(buf, "{ts}  {}{filename}{session_info}", r.project)?;
        if r.source == "slack" || r.turns.is_empty() || r.path.is_empty() {
            // Slack and .md files: use indexed bodies directly
            let idx = r.match_idx;
            let start = idx.saturating_sub(1);
            let end = (idx + 3).min(r.bodies.len());
            for i in start..end {
                let prefix = if i == idx { ">>>" } else { "   " };
                for line in r.bodies[i].lines().filter(|l| !l.is_empty()) {
                    writeln!(buf, "{prefix} {line}")?;
                }
            }
        } else if !r.session_id.is_empty() {
            // Claude/Codex: read turns via byte offsets
            let mi = if r.match_idx > 0 { r.match_idx - 1 } else { 0 };
            let ctx_start = mi.saturating_sub(1);
            let ctx_end = (mi + 2).min(r.turns.len());
            for i in ctx_start..ctx_end {
                let tm = &r.turns[i];
                let label = if tm.role == "user" {
                    "[User]"
                } else if r.source == "codex" {
                    "[Codex]"
                } else {
                    "[Claude]"
                };
                let prefix = if i == mi { ">>>" } else { "  " };
                writeln!(buf, "{prefix} {label} {}", format_date(&tm.timestamp))?;
                if let Some(turn) = read_turn_at(&r.path, &r.source, tm) {
                    for line in turn.text.lines().take(10) {
                        writeln!(buf, "{prefix}   {line}")?;
                    }
                }
            }
        }
    }
    if results.is_empty() {
        writeln!(buf, "no results")?;
    }
    std::io::stdout().write_all(&buf)?;
    Ok(())
}

fn format_date(iso: &str) -> String {
    let month = match iso.get(5..7) {
        Some("01") => "Jan",
        Some("02") => "Feb",
        Some("03") => "Mar",
        Some("04") => "Apr",
        Some("05") => "May",
        Some("06") => "Jun",
        Some("07") => "Jul",
        Some("08") => "Aug",
        Some("09") => "Sep",
        Some("10") => "Oct",
        Some("11") => "Nov",
        Some("12") => "Dec",
        _ => "???",
    };
    let day = iso.get(8..10).unwrap_or("??");
    let time = iso.get(11..16).unwrap_or("??:??");
    format!("{month} {day} {time}")
}

fn parse_range(s: &str) -> (usize, usize) {
    if let Some((a, b)) = s.split_once('-') {
        let start = a.parse().unwrap_or(0);
        let end = b.parse().unwrap_or(usize::MAX);
        (start, end)
    } else {
        let n = s.parse().unwrap_or(0);
        (n, n)
    }
}

fn dump(db_name: &PathBuf, session_id: &str, turns_range: Option<&str>, since_ms: Option<i64>) -> Result<()> {
    use witchcraft::types::*;
    use witchcraft::sql_generator::build_filter_sql_and_params;

    let db = DB::new_reader(db_name.clone()).unwrap();
    let name = session_id.strip_prefix('#').unwrap_or(session_id);

    // Try as session_id first, then thread conv_key, then channel_id/channel_name
    let (rows, is_channel) = {
        let (turn_start, turn_end) = turns_range.map(parse_range).unwrap_or((0, usize::MAX));
        let thread_ts = name.strip_prefix("thr:").unwrap_or(name);
        let thread_like = format!("%-{thread_ts}");

        // Helper: run a filter query and return matched rows
        fn query_filter(
            db: &DB,
            filter: &SqlStatementInternal,
            order: &str,
            turn_start: usize,
            turn_end: usize,
        ) -> Result<Vec<(String, String, i64)>> {
            let (filter_sql, params) = build_filter_sql_and_params(Some(filter))?;
            let sql = format!(
                "SELECT date, body, json_extract(metadata, '$.turn') as turn
                 FROM document WHERE {filter_sql} ORDER BY {order}"
            );
            let mut stmt = db.query(&sql)?;
            let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| &**p as &dyn rusqlite::ToSql).collect();
            let rows: Vec<(String, String, i64)> = stmt
                .query_map(param_refs.as_slice(), |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                })?
                .filter_map(|r| r.ok())
                .filter(|(_, _, t)| (*t as usize) >= turn_start && (*t as usize) <= turn_end)
                .collect();
            Ok(rows)
        }

        // 1. Try as Claude/Codex session_id
        let session_match = SqlStatementInternal {
            statement_type: SqlStatementType::Condition,
            condition: Some(SqlConditionInternal {
                key: "$.session_id".to_string(),
                operator: SqlOperator::Equals,
                value: Some(SqlValue::String(session_id.to_string())),
            }),
            logic: None,
            statements: None,
        };
        let rows = query_filter(&db, &session_match, "turn", turn_start, turn_end)?;
        if !rows.is_empty() {
            (rows, false)
        } else {
            // 2. Try as Slack thread TS (conv_key LIKE %-<ts>)
            let thread_match = SqlStatementInternal {
                statement_type: SqlStatementType::Condition,
                condition: Some(SqlConditionInternal {
                    key: "$.conv_key".to_string(),
                    operator: SqlOperator::Like,
                    value: Some(SqlValue::String(thread_like)),
                }),
                logic: None,
                statements: None,
            };
            let rows = query_filter(&db, &thread_match, "date", turn_start, turn_end)?;
            if !rows.is_empty() {
                (rows, true)
            } else {
            // 3. Try as channel_id/channel_name
            let channel_match = SqlStatementInternal {
                statement_type: SqlStatementType::Group,
                condition: None,
                logic: Some(SqlLogic::Or),
                statements: Some(vec![
                    SqlStatementInternal {
                        statement_type: SqlStatementType::Condition,
                        condition: Some(SqlConditionInternal {
                            key: "$.channel_id".to_string(),
                            operator: SqlOperator::Equals,
                            value: Some(SqlValue::String(name.to_string())),
                        }),
                        logic: None,
                        statements: None,
                    },
                    SqlStatementInternal {
                        statement_type: SqlStatementType::Condition,
                        condition: Some(SqlConditionInternal {
                            key: "$.channel_name".to_string(),
                            operator: SqlOperator::Equals,
                            value: Some(SqlValue::String(name.to_string())),
                        }),
                        logic: None,
                        statements: None,
                    },
                ]),
            };
            let since_filter = since_ms.map(|ms| {
                let cutoff_secs = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as i64
                    - ms / 1000;
                let cutoff_iso = chrono::DateTime::from_timestamp(cutoff_secs, 0)
                    .unwrap()
                    .to_rfc3339();
                SqlStatementInternal {
                    statement_type: SqlStatementType::Condition,
                    condition: Some(SqlConditionInternal {
                        key: "date".to_string(),
                        operator: SqlOperator::GreaterThanOrEquals,
                        value: Some(SqlValue::String(cutoff_iso)),
                    }),
                    logic: None,
                    statements: None,
                }
            });
            let filters: Vec<SqlStatementInternal> =
                [Some(channel_match), since_filter].into_iter().flatten().collect();
            let combined = if filters.len() == 1 {
                filters.into_iter().next().unwrap()
            } else {
                SqlStatementInternal {
                    statement_type: SqlStatementType::Group,
                    condition: None,
                    logic: Some(SqlLogic::And),
                    statements: Some(filters),
                }
            };
            let rows = query_filter(&db, &combined, "turn DESC", turn_start, turn_end)?;
            (rows, true)
        }}
    };

    if rows.is_empty() {
        eprintln!("No session or channel found for {session_id}");
        std::process::exit(1);
    }

    let mut buf = Vec::new();
    for (date, body, turn) in &rows {
        writeln!(buf, "---")?;
        if is_channel {
            writeln!(buf, "conv {}  {}", *turn, format_date(date))?;
        } else {
            writeln!(buf, "turn {}  {}", *turn, format_date(date))?;
        }
        for line in body.lines().skip_while(|l| {
            !is_channel && l.starts_with('[') && !l.starts_with("[User]") && !l.starts_with("[Claude]")
        }) {
            writeln!(buf, "{line}")?;
        }
    }
    if !buf.is_empty() {
        writeln!(buf, "---")?;
    }

    use std::io::IsTerminal;
    let output = String::from_utf8(buf)?;
    if std::io::stdout().is_terminal() {
        use std::process::{Command, Stdio};
        let mut pager = Command::new("less")
            .args(["-RFX"])
            .stdin(Stdio::piped())
            .spawn()?;
        pager.stdin.take().unwrap().write_all(output.as_bytes())?;
        let _ = pager.wait();
    } else {
        print!("{output}");
    }
    Ok(())
}

fn main() -> Result<()> {
    let _ = log::set_logger(&LOGGER).map(|()| log::set_max_level(LevelFilter::Warn));

    let args: Vec<String> = env::args().skip(1).collect();
    let mut session_filter: Option<String> = None;
    let mut exclude_sessions: Vec<String> = Vec::new();
    let mut since_ms: Option<i64> = None;
    let mut type_filter: Vec<String> = Vec::new();
    let mut num_results: Option<usize> = None;
    let mut dump_session: Option<String> = None;
    let mut turns_range: Option<String> = None;
    let mut dm_only = false;
    let mut no_dm = false;
    let mut unread_only = false;
    let mut current = false;
    let mut exclude_current = false;
    let mut query_args: Vec<&str> = Vec::new();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--nuke" => {
                let db_name = db_path();
                if db_name.exists() {
                    std::fs::remove_file(&db_name)?;
                    eprintln!("removed {}", db_name.display());
                } else {
                    eprintln!("no database to remove");
                }
                watermark::remove(&watermark::claude_path());
                watermark::remove(&watermark::codex_path());
                slack::remove_watermark();
                std::process::exit(0);
            }
            "--session" | "--channel" => {
                session_filter = iter.next().cloned();
            }
            "--exclude" => {
                if let Some(val) = iter.next() {
                    for id in val.split(',') {
                        let id = id.trim();
                        if !id.is_empty() {
                            exclude_sessions.push(id.to_string());
                        }
                    }
                }
            }
            "--since" => {
                if let Some(val) = iter.next() {
                    match parse_since(val) {
                        Some(ms) => since_ms = Some(ms),
                        None => {
                            eprintln!("invalid --since value: {val} (use e.g. 24h, 7d, 2w)");
                            std::process::exit(1);
                        }
                    }
                }
            }
            "--type" => {
                if let Some(val) = iter.next() {
                    for t in val.split(',') {
                        let t = t.trim();
                        if !t.is_empty() {
                            type_filter.push(t.to_string());
                        }
                    }
                }
            }
            "-n" => {
                if let Some(val) = iter.next() {
                    num_results = Some(val.parse().unwrap_or(0));
                }
            }
            "--dump" => {
                dump_session = iter.next().cloned();
            }
            "--turns" => {
                turns_range = iter.next().cloned();
            }
            "--current" => {
                current = true;
            }
            "--exclude-current" => {
                exclude_current = true;
            }
            "--dm" => {
                dm_only = true;
            }
            "--no-dm" => {
                no_dm = true;
            }
            "--unread" => {
                unread_only = true;
            }
            s if s.starts_with('-') => {
                eprintln!("unknown option: {s}");
                std::process::exit(1);
            }
            _ => {
                query_args.push(arg);
            }
        }
    }

    use std::io::IsTerminal;
    if std::io::stderr().is_terminal() {
        eprintln!("pickbrain {} — Copyright (c) 2026 Dropbox Inc.", env!("CARGO_PKG_VERSION"));
    }

    let db_name = db_path();
    let assets = assets_path();

    // Migrate DB from old location (~/.claude/pickbrain.db)
    if !db_name.exists() {
        let home = env::var("HOME").unwrap_or_default();
        let old_db = PathBuf::from(home).join(".claude/pickbrain.db");
        if old_db.exists() {
            eprintln!("migrating database from {} to {}", old_db.display(), db_name.display());
            std::fs::rename(&old_db, &db_name).ok();
        }
    }

    // Detect the calling session once — used for both ingest-skip and --current filter.
    let active_session = detect_active_session();

    if current || exclude_current {
        match &active_session {
            Some(id) => {
                if current {
                    session_filter = Some(id.clone());
                }
                if exclude_current {
                    exclude_sessions.push(id.clone());
                }
            }
            None => {
                let flag = if current { "--current" } else { "--exclude-current" };
                eprintln!("{flag}: could not detect active session");
                std::process::exit(1);
            }
        }
    }

    // Skip the active session's JSONL if its watermark is fresh (<10 min).
    // If we can't detect the active session, nothing is skipped (eager by default).
    let stale_ms = 10 * 60 * 1000;
    match ingest(&db_name, active_session.as_deref(), stale_ms, &type_filter) {
        Ok(have_changes) => {
            if have_changes {
                let db_rw = DB::new(db_name.clone()).unwrap();
                let device = witchcraft::make_device();
                let embedder = witchcraft::Embedder::new(&device, &assets)?;
                embed_and_index(&db_rw, &embedder, &device)?;
            }
        },
        Err(e) => {
            eprintln!("warning: ingest failed: {e}");
            std::process::exit(1);
        }
    }

    let has_filters = session_filter.is_some() || !exclude_sessions.is_empty() || since_ms.is_some()
        || !type_filter.is_empty() || dm_only || no_dm || unread_only || current || exclude_current;

    if let Some(ref sid) = dump_session {
        dump(&db_name, sid, turns_range.as_deref(), since_ms)?;
    } else if !query_args.is_empty() || has_filters {
        let q = query_args.join(" ");
        if std::io::stdout().is_terminal() {
            let n = num_results.unwrap_or(0);
            if let Some((sid, path, source)) = search_tui(&db_name, &assets, &q, session_filter.as_deref(), &exclude_sessions, since_ms, &type_filter, n, dm_only, no_dm, unread_only)? {
                launch_resume(&sid, &path, &source)?;
            }
        } else {
            let n = num_results.unwrap_or(20);
            search_plain(&db_name, &assets, &q, session_filter.as_deref(), &exclude_sessions, since_ms, &type_filter, n, dm_only, no_dm, unread_only)?;
        }
    } else {
        eprintln!("Usage: pickbrain [options] <query>");
        eprintln!("       pickbrain --dump <session-id|channel|thr:ts> [--turns N-M] [--since T]");
        eprintln!("       pickbrain --nuke");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --session ID         search within a session, channel, or thread");
        eprintln!("  --current            search within the calling session");
        eprintln!("  --exclude UUID,...   exclude sessions from results");
        eprintln!("  --exclude-current    exclude the calling session");
        eprintln!("  --since 24h|7d|2w    only search recent history");
        eprintln!("  --type claude,slack  filter by source (claude, codex, slack)");
        eprintln!("  -n N                 number of results (0=unlimited, default: unlimited in TUI, 20 in pipe)");
        eprintln!("  --dm                 only DMs (Slack)");
        eprintln!("  --no-dm              exclude DMs (Slack)");
        eprintln!("  --unread             only unread conversations (Slack)");
        eprintln!("  --dump ID            dump a session, channel, or thread");
        eprintln!("  --turns N-M          limit turns/conversations in dump");
    }
    Ok(())
}
