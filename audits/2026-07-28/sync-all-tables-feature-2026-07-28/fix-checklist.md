# Fix checklist — sync-all fan-out

**Parent audit:** [audit.md](./audit.md)

**Fix flow (trees):** skipped (single P0 item)

**Post-completion:** deferred

**Goal:** One Make/CLI command fans out pairwise sync from a source env to every non-protected target.

**Primary code defect:** CLI/Makefile expose only pairwise sync; no loop over `list_environments()` — `db_management.py` sync CLI + `Makefile` sync targets.

**Patch scope:** One P0 item — thin fan-out only (no `--table` filter, no DDL broadcast).

**Prerequisites:** None.

---

## Scope gate

- [x] Checklist implements only `P0-code` rows from audit Resolution table (fan-out wrapper + CLI/Make; defer `--table`)
- [x] Contributing factors tiered EPIC/ops — not copied as implementation items
- [x] Single P0 item → skip fix-flow

---

## Implementation order

| # | Item | Tier | Audit § | Primary files |
|---|------|------|---------|---------------|
| 1 | `sync-all` / `sync-all-apply` fan-out | P0-code | Resolution + suggested defaults | `db_management.py`, `Makefile`, `__init__.py`, `README.md`, tests |

---

## 1. sync-all fan-out

**Problem:** Operators must repeat `sync-apply` per target environment.

**Tasks**

- [x] Add `sync_all_environments_from_source` looping `list_environments(exclude_prod=True)`, skip source, call `sync_environment_from_source` per target
- [x] CLI `sync-all` (`--apply` required for writes; dry-run default) + Make `sync-all` / `sync-all-apply`
- [x] Export from `dbase.database`; document in README
- [x] Unit tests: skips source, skips protected via list_environments, dry-run vs apply

**Acceptance**

- [x] `make sync-all SOURCE_ENV=…` dry-runs against all non-protected targets except source
- [x] `make sync-all-apply SOURCE_ENV=…` applies; optional `WITH_DATA=1` / `BRANCH=…`
- [x] Reuses pairwise sync semantics (`sync_databases=True`, `sync_tables=True`)
- [x] Tests pass without live MySQL

**Audit trace:** [audit.md](./audit.md)

---

## Test plan

1. [x] Unit: dry-run calls sync once per target with `apply=False`
2. [x] Unit: apply path passes `apply=True`
3. [x] Unit: source env excluded from targets
4. [x] `make -C dbase/database help` shows new targets

---

## Out of scope (EPIC / redesign)

- `--table` / `--base-name` filters
- DDL broadcast (create table from SQL file)
- `sync_databases=False` default for add-table-only UX
- Parallel apply; column/ALTER sync; `--include-protected`

---

## Progress log

| Date | Items done | Notes |
|------|------------|-------|
| 2026-07-28 | #1 | `sync_all_environments_from_source` + CLI/Make/README/tests; 4 unit tests passed |
