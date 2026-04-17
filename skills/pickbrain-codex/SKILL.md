---
name: pickbrain
description: Semantic search over past Claude Code and Codex conversations and memories. Use when the user wants to recall, find, or reference something from a previous coding session — e.g. "what did we discuss about X", "find that conversation where we fixed Y", "search my history for Z".
---

# Pickbrain — Semantic Search for AI Coding History

Search past Claude Code and Codex conversations, memory files, and authored files using semantic search.

## Usage

Run `pickbrain` with the user's query:

```bash
pickbrain "$ARGUMENTS"
```

Pickbrain automatically ingests new sessions/memories before each search.

Use `pickbrain --update` (without a query) to force a full ingest+embed pass.

## Interpreting Results

Each result includes:
- **Timestamp** and **project directory**
- **Session ID** and **turn number** — identifies the exact conversation turn
- **Matching text** — the relevant chunk from the conversation

Present results as a concise summary. Quote the most relevant excerpts. To dig deeper into a specific session:

```bash
pickbrain --dump <session-id> --turns <start>-<end>
```

## Filtering

Search within a specific session:

```bash
pickbrain --session <session-id> "<query>"
```

Exclude specific sessions from results (comma-separated or repeated):

```bash
pickbrain --exclude <uuid1>,<uuid2> "<query>"
pickbrain --exclude <uuid1> --exclude <uuid2> "<query>"
```

## Notes

- First run requires a full ingest+embed pass (~7s). Subsequent searches are incremental.
- The database lives at `~/.claude/pickbrain.db`.
- Results are ranked by semantic similarity — they may not contain the exact query words.
