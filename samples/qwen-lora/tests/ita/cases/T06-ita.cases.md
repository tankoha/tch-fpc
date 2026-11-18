| Case ID       | Title                                                      | Expected (short)                                      |
|---------------|------------------------------------------------------------|-------------------------------------------------------|
| ITa-T06-001   | Single objfpc file, example, exact output match           | compiled-run, accepted                                |
| ITa-T06-002   | Single objfpc file, example, no expected file             | compiled-run, no-expected                             |
| ITa-T06-004   | Single objfpc file, expect-error                           | compiled-run, {"compiled": false, "match": "expect-error"} |
| ITa-T06-006   | Single fpc-delphi file, example, exact match               | compiled-run, accepted                                |
| ITa-T06-007   | Single unleashed file, example, exact match                | compiled-run, accepted (with QLORA_FPC_UNLEASHED)     |
| ITa-T06-008   | Single oxygene file, static-ok                             | static-only, static-ok                                |
| ITa-T06-009   | Single oxygene file, static-reject                         | static-only, static-reject                            |
| ITa-T06-010   | Directory with mixed dialects (examples and pitfalls)      | All files: dialect/class per facts table              |
| ITa-T06-011   | Timeout handling (example)                                 | compiled-run, timeout                                 |
| ITa-T06-012   | Timeout handling (compile-time)                            | compiled-run, skipped, timeout                        |
| ITa-T06-013   | Oracle run (each skill directory, one dialect)             | One line per file, per skill directory                |