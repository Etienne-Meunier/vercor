# VerCOR Progress Archive: 2026-07-22

This archive preserves completed July 22 work moved out of the bounded active
orientation log in `PROGRESS.md`.

## Completed work

- Cadence-aware period filename precision implemented locally: daily, monthly,
  and yearly means use `YYYY-MM-DD`, `YYYY-MM`, and `YYYY`; step filenames and
  exact NetCDF window-start metadata remain unchanged. TDD RED was 7 failed/2
  passed, GREEN 9/9, output focus 17/17, fast 660/660, and full/coverage
  1,266/1,266 at 91%. Black left 234 files unchanged (known Python 3.13
  safety-parse advisory), flake8 reported 0, mypy passed 234 source files, and
  compileall/whitespace passed; the Veros fake-state coverage fixture supplies
  the minimal empty settings field tuple.
- Veros payload copy compatibility fixed locally: both `copy_state()` paths
  freeze the native settings `dict_keys` view, preserving generic payload
  ownership. RED/GREEN was 2/2; affected tests passed 81/81, runnable fast
  passed 656/656, real preparation and static gates passed. Apple Git later
  became available and the remaining repository policy test passed.
