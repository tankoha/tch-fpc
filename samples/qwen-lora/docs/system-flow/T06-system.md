# System Flow for T06: `DialectVerify` and `qlora_verify`

```mermaid
flowchart TD
    start([Start]) --> args{Parse CLI args}
    args -->|FAIL| usage[Print usage
exit 2]
    args -->|OK| gather[Gather files: *.pas, *.dpr
under PATHs, sorted]

    gather --> loop{For each file}
    loop -->|done| exit_code[Exit 0 if all passed/skipped
else 1]

    loop --> process[Detect or force dialect]
    process --> class{Class}
    class -->|static-only| oxy[Static check: first word = namespace?]
    class -->|compiled-run/fpc-approx| compile[Run compiler
with dialect flags]

    oxy --> outcome_static[Outcome: static-ok or static-reject]
    outcome_static --> write[Write TSV line]
    write --> loop

    compile -->|FAIL| skipped[Outcome: skipped
(detail: why)]
    compile -->|OK| run[Run program
with --timeout]

    run -->|FAIL| outcome_run[Outcome: run-error, timeout or skipped]
    run -->|OK| compare[Compare stdout
to .expected file]

    compare --> expect_error{expect-error?}
    expect_error -->|yes| outcome_expect[Outcome: accepted if compiler output
contains <text>, else wrong-error]
    expect_error -->|no| outcome_compare[Outcome: accepted if match, else wrong-output
or no-expected]

    skipped --> write
    outcome_run --> write
    outcome_expect --> write
    outcome_compare --> write
```