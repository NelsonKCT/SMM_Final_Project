<!-- copied from AI-Harness 2026-07-04; re-run /adopt-project to refresh -->

# Git workflow & safety

- Never commit directly to `main`/`master` in a repo that has a remote; branch as
  `feature/<topic>` or `fix/<topic>` first.
- Commit subjects: imperative, <72 chars, conventional prefix when natural
  (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, `test:`).
- Before every commit, review the staged diff for secrets: `.env*`, API keys,
  tokens, credentials, personal data. If found, stop and tell the user.
- Never run destructive git commands (`push --force`, `reset --hard`,
  `clean -fd`, branch deletion) without explicit user confirmation in this session.
- Commit only when the user asks. Never push unless the user asks.
