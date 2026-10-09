# Dataset row-selection correction (2026-10-03)

This note records a correction to the selection of the earliest post-treatment resting-state observation in the DishBrain analysis. It is part of the project provenance and supersedes any earlier result that used a later-session value to fill a missing value in the first session.

## What changed

The previous implementation used Pandas `GroupBy.first()`. That operation selects the first non-null value separately for each column, so the resulting feature row need not correspond to one actual recording session. If a feature was missing in the earliest rest session, it could be filled from a later rest session.

The corrected implementation sorts by session number and keeps the first complete row for each experiment key. Missing values in that earliest row remain missing and are handled by the downstream model pipeline as missing values; they are not backfilled from later sessions.

## Affected values

Four earliest-rest `DCC` values that had previously been taken from later sessions are now missing: chips 14218 (2022-04-27), 14229 (2022-04-29), 14236 (2022-05-04), and 15812 (2022-05-20). For chip 14236 on 2022-05-04, two other values from the same later row are also now missing: `Shape Collapse beta` and `Shape Collapse error`.

The `DCC` missingness in the 112-row experiment table is now 30/112 (26.79%), rather than 23.21% in the pre-correction table.

## Consequences for reported results

- Full-cohort, chip-grouped ExtraTrees OOF balanced accuracy remains 0.737477 (0.7375 rounded); macro-F1 and macro OVR AUC are 0.715508 and 0.879350.
- The shared 11-chip / 64-record electrophysiology subset changes to balanced accuracy 0.689394 and macro-F1 0.692788. The earlier 0.719697 / 0.724377 values are superseded.
- Some response-regression and retrospective candidate-portfolio results also change; use only values regenerated from the corrected table.
- Order-only and dose-only metadata diagnostics do not use the affected electrophysiology features; they remain post-hoc risk diagnostics, not fair model comparisons or causal evidence.

## Reproduction and limitations

Run `src/audit_dishbrain.py`, then the analyses listed in `README.md`, followed by `python -m unittest discover -s tests -v`. The regression test `tests/test_earliest_rest_semantics.py` verifies that a later non-null feature cannot fill the earliest session's missing value.

This correction changes the selected baseline row, not the original source files. It does not provide independent laboratory validation, establish drug-specific causal effects, or resolve source-data licensing. The source dataset should be obtained from its original repository according to `DATA_ACCESS_AND_REPRODUCTION.md`.
