# UT cases — unit `EvalReport` (T20, stage 1; task T20b1)

Every row has one test method; every test method belongs to one row.
Expected values of numeric cases are taken from the case table as-is (the
facts give exact numbers / fractions); error rows compare the whole
message.

| Case ID | Title | Target function | Input (short) | Expected result | Notes |
|---|---|---|---|---|---|
| UT-EvalReport-001 | CountOf, every column | CountOf | `CountOf` of a, b, c objfpc (passed, compiled only, neither), d fpc-delphi, e delphi, f unleashed (failed), g oxygene, a objfpc thinking off, regression t, u (tool-call, passed, failed), w (contains, passed); columns objfpc, verified, delphi, tool-call, contains (thinking on), verified (off), `pascal` | 3/2/1 (N/compiled/passed); 6/4/3; N 1; N 2, passed 1; N 1; N 1; N 0 | |
| UT-EvalReport-002 | FlipsOf, pairing by task and seed | FlipsOf | (a1 = task a, seed 1) before a1 (fail), a2 (pass), b1 (pass), c1 (fail), d1 (fail, no partner), a1 off (fail), x1 delphi (fail); after a1 (pass), a2 (fail), b1, c1 (pass), a1 off (pass), x1 (pass): `FlipsOf` objfpc on, verified on, objfpc off, delphi on | N 4, B 2, C 1; N 4; N 1, B 1; B 1 | |
| UT-EvalReport-003 | SignTestP, exact two-sided | SignTestP | `SignTestP` (0,0), (5,0), (0,6), (10,2), (2,10), (3,3), (39,1) | 1; 0.0625; 0.03125; 158/4096 (twice); 1; 2 · 41 / 2^40 | |
| UT-EvalReport-004 | PassRate | PassRate | `PassRate` of N 0; of N 4, passed 3 | NaN; 0.75 | |
| UT-EvalReport-005 | Adoption, adopt: yes | Adoption | before `Series('objfpc', 10, 2)` + a passed tool call; after `Series('objfpc', 10, 9)` + the same; ratio 0.8 | `verified aggregate rises (0.200 -> 0.900): yes`, `flips significant (7 fail->pass, 0 pass->fail, p = 0.0156): yes`, `no verified dialect falls by more than 5 points: yes`, `tool-call check loses nothing (1 -> 1): yes`, `H49 speed at least 0.7 of before (0.800): yes`, `adopt: yes` | |
| UT-EvalReport-006 | Adoption, adopt: no | Adoption | before objfpc 10/2 + fpc-delphi 10/10 + tool call passed; after objfpc 10/10 + fpc-delphi 10/9 + tool call failed; ratio 0.65 | `.. (0.600 -> 0.950): yes`, `.. (8 fail->pass, 1 pass->fail, p = 0.0391): yes`, `no verified dialect falls by more than 5 points fpc-delphi: no`, `tool-call check loses nothing (1 -> 0): no`, `H49 .. (0.650): no`, `adopt: no` | |
| UT-EvalReport-007 | Adoption, undecided; empty mode | Adoption | objfpc 10/2 → 10/9 without tool calls; ratio NaN; then thinking off (empty), ratio 1 | line 2 `flips significant (7 fail->pass, 0 pass->fail, p = 0.0156): yes`, `.. (0 -> 0): yes`, `H49 speed at least 0.7 of before: not measured`, `adopt: undecided`; line 6 `adopt: no` | |
| UT-EvalReport-008 | DialectsOf | DialectsOf | `DialectsOf` of unleashed, objfpc, regression, delphi, objfpc; of nil | `delphi\|objfpc\|unleashed`; none | |
| UT-EvalReport-009 | FirstBlock | FirstBlock | `FirstBlock` of `text`, a block `a`/`  b` closed by ```` ```  ````, then a block `c`; of an unclosed block; of `no block`; of ```` ``` ````#10```` ``` ```` | True, `a`#10`  b`#10; False, ''; False; True, '' | |
