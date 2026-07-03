# Project Instructions

This file defines the shared working rules for all chats and agents operating in this repository.

## Session Start

Read these files at the start of every session:

1. `PROJECT_INSTRUCTIONS.md`
2. `PROJECT_CONTEXT.md`
3. `PROJECT_OUTLINE.md`
4. `PROJECT_PROGRESS.md`
5. `SESSION_SUMMARY.md`

Use `CODEX_CONTEXT.md` as the detailed long-form design brief when needed.

## Session End

Before ending a session:

1. Update `PROJECT_PROGRESS.md` if the repository state or next steps changed.
2. Update `SESSION_SUMMARY.md` with a concise handoff.
3. Note the current branch, key files changed, verification run, and immediate next action.
4. If significant working changes were made, commit them before ending the session unless the user explicitly says not to.

## File Ownership Rules

- `PROJECT_OUTLINE.md` is the stable project brief. Do not add session chatter there.
- `PROJECT_PROGRESS.md` is the working-state tracker. Keep it current and concise.
- `SESSION_SUMMARY.md` is the handoff note for the next session. It should be readable in under a minute.
- `CODEX_CONTEXT.md` is the detailed design reference. Update it only when the underlying project brief changes materially.

## Git Workflow

- Never commit directly to `main` or `master`.
- Always create a branch named `type/short-description`.
- Keep branches scoped to one logical feature, fix, refactor, or docs change.
- Make small atomic commits with messages like `feat: ...`, `fix: ...`, `docs: ...`, or `chore: ...`.
- Commit every significant working change so reverting is easy.
- Prefer several small working commits during implementation over one large commit at the end of a session.
- Do not merge to `main` without explicit approval.

## Multi-Agent Coordination

- Assume another chat or agent may read these files immediately after this session.
- Leave clear notes instead of relying on implicit memory or terminal history.
- If you intentionally leave stubs, placeholders, or partial work, say so explicitly in `PROJECT_PROGRESS.md` and `SESSION_SUMMARY.md`.
