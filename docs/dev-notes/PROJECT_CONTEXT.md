# Project Context

This file is the session index for humans, chats, and coding agents working in this repository.

## Read Order

Read these files at the start of every session:

1. `PROJECT_INSTRUCTIONS.md`
2. `PROJECT_CONTEXT.md`
3. `PROJECT_OUTLINE.md`
4. `PROJECT_PROGRESS.md`
5. `SESSION_SUMMARY.md`

Read `CODEX_CONTEXT.md` when detailed architectural or research context is needed beyond the short outline.

## File Roles

- `PROJECT_INSTRUCTIONS.md`: working rules, git workflow, and file update expectations
- `PROJECT_OUTLINE.md`: stable project goals, scope, architecture, and staged roadmap
- `PROJECT_PROGRESS.md`: current state of the repository, latest completed work, and next priorities
- `SESSION_SUMMARY.md`: concise end-of-session handoff for the next chat or agent
- `CODEX_CONTEXT.md`: detailed archival design brief

## Update Rules

- Update `PROJECT_OUTLINE.md` only when the scope, architecture, or staged plan changes.
- Update `PROJECT_PROGRESS.md` whenever milestones, current status, or next steps change.
- Update `SESSION_SUMMARY.md` at the end of every working session.
- Update `PROJECT_INSTRUCTIONS.md` when the collaboration process or workflow rules change.

## Coordination Notes

- Assume multiple chats or agents may touch the repo in parallel or in close succession.
- Keep handoff notes explicit about branch, files changed, verification run, and next recommended action.
- Do not treat `SESSION_SUMMARY.md` as a full design document; keep it short and operational.
- Play a local completion sound at the end of each finished task so the user gets an audible cue.

## Remote HPC Workflow

- Verified against the Bowdoin HPC knowledge-base docs on July 3, 2026.
- Preferred terminal automation path: SSH from macOS or Linux to `moosehead.bowdoin.edu`.
- Human fallback: the Bowdoin HPC Web Portal at `hpcweb.bowdoin.edu` in Firefox or Chrome. Its shell also lands on `moosehead.bowdoin.edu`.
- Off-campus access requires the Bowdoin VPN for SSH, the web portal, JupyterLab, and RStudio.
- Optional short interactive host: `dover.bowdoin.edu`.
- Treat the local machine as the editing and orchestration environment and Bowdoin HPC as the preferred execution environment for upstream LivePortrait installs and longer runs.
- Long GPU jobs should go through Slurm on the `gpu` partition with explicit `--gres`. The preferred 96 GB target is `--gres=gpu:pro6000:1` when available.
- Never commit Bowdoin credentials. If password auth is needed, keep it only in a local ignored file such as `.env.hpc.local` or in the system keychain. Prefer SSH key auth if Bowdoin allows it for this account.
- Verified on July 3, 2026: password-based SSH to `moosehead.bowdoin.edu` works from this machine using the local `.env.hpc.local` file and `expect`.
- Local reusable Codex skill path: `~/.codex/skills/bowdoin-hpc-ssh`. Use it for repeatable Bowdoin SSH commands in future sessions.
- Verified remote code workspace: `/home/kelsedfy/video-persona-gen`, with upstream LivePortrait staged at `/home/kelsedfy/video-persona-gen/external/LivePortrait`.
- `moosehead` is a Slurm headnode for git, file, and job-submission work. Do not run Python workloads there directly; use Slurm or an approved interactive machine instead.
- Verified remote LivePortrait environment path: `/home/kelsedfy/video-persona-gen/.conda/liveportrait`.
- Bowdoin home storage is still at its hard quota, so direct large writes into `/home/kelsedfy` remain unsafe.
- Verified on July 4, 2026: `/mnt/hpc/tmp/kelsedfy` is writable, has about `32T` free on the backing Gluster filesystem, and is the current durable scratch root for this project.
- The tracked Bowdoin jobs now use `/mnt/hpc/tmp/<user>/video-persona-gen` for persistent weights, logs, and outputs while still staging runtime work under node-local `/tmp`.
