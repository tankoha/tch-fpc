# T22 System Flow: Oxygene in the Verification Pipeline

## Users of the Oxygene Filter

### qlora_dataset (T19)

`qlora_dataset` uses `qlora_verify`'s filter on the Oxygene records of M3:
the generated dataset. Each Oxygene source file is compiled and run to
produce a `compiled-run` class outcome (or `compile-error`, `run-error`,
etc.). The dataset stores source, dialect, class, outcome, and detail.

### qlora_eval (T20)

`qlora_eval` uses the filter on Oxygene replies from the model being
evaluated. Replies considered valid Oxygene are compiled and run; the
outcome determines whether the reply is accepted (matches expected output) or
rejected (compile error, wrong output, etc.).

### Evaluation Tasks' Reference Solutions

The evaluation tasks (M3) include Oxygene reference solutions. These are
verified with the same filter to ensure they compile and produce correct
output before being used as expected outputs for model evaluation.

## Unit Tests

The UT runs EBuild and dotnet inside the flatpak. The Oxygene test rows
(UT-DialectVerify-301 to 305) verify:

- Correct extraction of `Elements` and `OxyRoot` from `ReadEnvRec`
- EBuild error line recognition (`E: ` prefix)
- `OxygeneStep` success, failure, and timeout paths
- `RemoveTree` recursive deletion
- Full `VerifyFile` flow for Oxygene: accepted, compile-error, run-error,
  wrong-output, no-expected, accepted with pitfall, wrong-error

The tests use paths under `work/ut/` for the Oxygene work directory to avoid
interfering with the user's `~/.cache/qlora-verify/`.

## ITa

The ITa runs in two parts:

1. `tests/ita/T22-run.sh` verifies `tests/ita/T22/files` (correct Oxygene
   programs and planted faults) with `--dialect oxygene`. Checks that:
   - Outcomes match planted values
   - Details are free of home and work directory paths
   - No work directories left under `~/.cache/qlora-verify`
   - Build and run time per file is under 5 seconds

2. The T19 ITa runs again with the updated `src/`. Now the raw record
   `x-oxygene` has a planted wrong output, testing that Oxygene sources in
   the dataset pipeline are compiled and rejected correctly.

Both parts use the rehearsal GGUF model (read from `QLORA_MODEL` environment
variable) and write results to `work/ita/T22/summary.tsv`.
