---
name: pickbrain
description: Semantic search over past Claude Code conversations and memories. Use when the user wants to recall, find, or reference something from a previous Claude session — e.g. "what did we discuss about X", "find that conversation where we fixed Y", "search my Claude history for Z".
allowed-tools: Bash
---

# Pickbrain — Semantic Search for Claude History

Search past Claude Code conversations, memory files, and authored files using semantic search.

## Installation

Build and install from the repo root:

```bash
make pickbrain
# then copy or symlink the binary to somewhere on your PATH:
ln -sf "$(pwd)/pickbrain" ~/bin/pickbrain
```

The binary embeds its own model weights (`embed-assets` feature), so no separate assets directory is needed.

## Usage

Run `pickbrain` via Bash with the user's query:

```bash
pickbrain "$ARGUMENTS"
```

Pickbrain automatically ingests new sessions/memories before each search (cheap filesystem walk).
New chunks are embedded using the same model load as the search query, so there is no extra startup cost.

Use `pickbrain --update` (without a query) to force a full ingest+embed pass.

## Interpreting Results

Each result includes:
- **Timestamp** and **project directory**
- **Session ID** and **turn number** — identifies the exact conversation turn
- **Matching text** — the relevant chunk from the conversation

Present results as a concise summary. Quote the most relevant excerpts. If the user wants to dig deeper into a specific session, use `--dump`:

```bash
pickbrain --dump <session-id> --turns <start>-<end>
```

## Filtering

To search within a specific session:

```bash
pickbrain --session <session-id> "<query>"
```

To exclude specific sessions from results (comma-separated or repeated):

```bash
pickbrain --exclude <uuid1>,<uuid2> "<query>"
pickbrain --exclude <uuid1> --exclude <uuid2> "<query>"
```

## Notes

- First run requires a full ingest+embed pass (~7s). Subsequent searches auto-ingest incrementally.
- The database lives at `~/.claude/pickbrain.db`.
- Results are ranked by semantic similarity — they may not contain the exact query words.
