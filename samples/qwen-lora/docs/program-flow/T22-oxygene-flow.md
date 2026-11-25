# Program flow — Oxygene build and run in the filter (T22)

Step-by-step control flow of the T22 changes in the unit `DialectVerify` (T06): `ReadEnvRec`
(the Oxygene part), the Oxygene branch of `ChooseCompiler`, `FirstCompilerErrorLine`,
`OxygeneStep`, `RemoveTree` and `VerifyFile` (both the Oxygene and the non-Oxygene path). The
exact code is in `docs/detailed-design/T22-detailed.md` (§2–8) with the outcome table in §9;
the external design is `docs/external-design/T22-external.md`; the facts are
`tasks/T22-facts.md`. The T06 functions that stay unchanged are `DetectDialect`, `CompileStep`,
`ExtractExpectErrorText`, `ExpectedPathFor`, `RunLimited`, `ReadTextFile`, `NormalizeDetail`,
`IsGoodOutcome`. A build is good only when EBuild exits 0 **and** `Bin/Ex.dll` exists
(`T22-detailed.md` §6). Stages (T22-facts): stage 1 (T22b) is the implementation with rows
`UT-DialectVerify-013`, `103`, `104` (changed) and `301` to `305`; T22c is the ITa
(`ITa-T22-001` to `006`).

## `ReadEnvRec` flow (the Oxygene part)

```mermaid
flowchart TD
    A([T06 part — Fpc from AFpc or default fpc; FpcUnleashed from AFpcUnleashed or the default of AHome]) --> E[Result.Elements := com.remobjects.earth — the flatpak app id, a fixed constant]
    E --> O[Result.OxyRoot := AHome plus /.cache/qlora-verify — the Oxygene work root, under the home directory]
    O --> Out([Return the record — the work dir under home: the flatpak has a /tmp of its own])
```

`UT-DialectVerify-301`: `ReadEnvRec('', '', '/h')` → `Elements` `com.remobjects.earth`,
`OxyRoot` `/h/.cache/qlora-verify`.

## `ChooseCompiler` flow (the Oxygene branch)

```mermaid
flowchart TD
    A([ChooseCompiler D, AEnv, AClass — the other arms are T06, unchanged]) --> X{D is D_OXYGENE?}
    X -->|no| OT[objfpc / fpc-delphi / unleashed / delphi arms — T06]
    X -->|yes| FP[Result := ExeSearch flatpak — the flatpak executable from PATH; empty when absent]
    FP --> CL[AClass := compiled-run — like the other compiled dialects; T06 gave static-only]
    CL --> Out([Return the path and the class])
```

`UT-DialectVerify-013` (changed): `ChooseCompiler('oxygene', ReadEnvRec('', '', '/h'), Cls)`
→ `ExeSearch('flatpak')`, `Cls` `compiled-run`. An empty path flows through `OxygeneStep` to
`csNoCompiler` → `skipped` / `compiler not found`.

## `FirstCompilerErrorLine` flow (the `E: ` prefix)

```mermaid
flowchart TD
    A([each line of the log, first match wins]) --> C1{line contains error: or fatal: — the T06 test, lowercased}
    C1 -->|yes| R[Result := the line]
    C1 -->|no| C2{line begins exactly with E, colon, space — the T22 test, positional and case-sensitive}
    C2 -->|yes — EBuild error line, e.g. E: Unknown identifier Foo, file Program.pas line 7| R
    C2 -->|no — XE: not this, indented E:, lowercase e:| NX[next line; none left and no match: Result stays empty]
```

`UT-DialectVerify-302`: of the lines `Building`, `XE: not this`,
`E: Unknown identifier "Foo" [Program.pas (7)]` → the `E: ` line; of ` E: indented`,
`e: lower` → `''`.

## `OxygeneStep` flow

```mermaid
flowchart TD
    A([OxygeneStep AFlatpak, AApp, ASrcPath, AOutDir, ATimeoutMs, ALog]) --> Z[ALog := empty]
    Z --> CP{AFlatpak empty, or not IsPathUsable?}
    CP -->|yes| NC([Exit csNoCompiler — no flatpak, later: skipped / compiler not found])
    CP -->|no| FD[ForceDirectories AOutDir — the work dir under OxyRoot]
    FD --> CPY[read ASrcPath; write AOutdir/Program.pas — one copy of the source]
    CPY --> PRJ[write AOutdir/Ex.elements from K_OXY_PROJECT — the .NET Core, Echoes project]
    PRJ --> EB[RunLimited in AOutdir: flatpak run, command /app/dotnet/dotnet, AApp, /app/earth/EBuild.dll, Ex.elements — the pascal-oxygene skill line]
    EB --> RP[ALog := Out with AOutdir/slash replaced by nothing — no home path in a detail]
    RP --> TO{TimedOut?}
    TO -->|yes| CT([Exit csTimeout — later: skipped / timeout])
    TO -->|no| OK{Code is 0 and AOutdir/Bin/Ex.dll exists?}
    OK -->|no — either one missing| CF([Exit csFail — later: compile-error or wrong-error])
    OK -->|yes| GO([Return csOk])
```

`UT-DialectVerify-303`: the good source into `<OxyRoot>/a` → `csOk`, and `a` holds
`Program.pas`, `Ex.elements`, `Bin/Ex.dll`; the bad one into `/b` → `csFail`,
`FirstCompilerErrorLine(Log)` = `E: Unknown identifier "Foo" [Program.pas (8)]`, the directory
not in `Log`; `''` as the flatpak → `csNoCompiler`; `/nonexistent/flatpak` → `csNoCompiler`;
the good source with a 1 ms limit → `csTimeout`.

## `RemoveTree` flow

```mermaid
flowchart TD
    A([RemoveTree ADir]) --> F1{FindFirst ADir/star, files and directories — found any?}
    F1 -->|no| RM[RemoveDir ADir]
    F1 -->|yes| DN{entry is dot or dot dot?}
    DN -->|yes| NX[FindNext]
    DN -->|no| ISD{entry attribute is faDirectory?}
    ISD -->|yes| RC[RemoveTree ADir/slash/entry — recurse into the subtree]
    ISD -->|no| DF[DeleteFile ADir/slash/entry]
    RC --> NX
    DF --> NX
    NX --> DN
    NX -->|end of entries| RM
    RM --> Out([Return — ADir and everything below it is gone])
```

Bottom-up: subtrees first, files, then `ADir` itself; nothing below the parent is touched.
`UT-DialectVerify-304`: `t/a.txt`, `t/s/b.txt`, `t/s/u/c.txt` under it — after
`RemoveTree('.../t')`, `t` is gone and the parent (`work/ut/dv-304`) is still there.

## `VerifyFile` flow (both paths)

```mermaid
flowchart TD
    A([VerifyFile ASrcName, AForcedDialect, ATimeoutSec, AEnv, AWorkRoot, AOK]) --> RT[ReadTextFile ASrcName — missing file: empty text, classified, never raised]
    RT --> DL{AForcedDialect non-empty, lowercased?}
    DL -->|yes| DC[Dialect := the forced dialect]
    DL -->|no| DD[Dialect := DetectDialect of text and name — T06]
    DC --> CC[ChooseCompiler Dialect, AEnv — Comp and CClass]
    DD --> CC
    CC --> SQ[GVerifySeq := GVerifySeq plus 1 — n unique per call, no collision with a stale dir]
    SQ --> ISX{Dialect is D_OXYGENE?}
    ISX -->|yes — the Oxygene path| OW[W := AEnv.OxyRoot/slash/n — the work dir under the home directory; AWorkRoot not used at all]
    OW --> OB[State := OxygeneStep Comp, AEnv.Elements, ASrcName, W, ATimeoutSec times 1000 — build with EBuild in the flatpak]
    ISX -->|no — the T06 path| NW[W := AWorkRoot/slash/n]
    NW --> NB[State := CompileStep Comp, ASrcName, W, ATimeoutSec times 1000, delphi flag, ExePath — T06 flags, delphi copy]
    OB --> ST{State of the build}
    NB --> ST
    ST -->|csNoCompiler or csStartFail| SK1[Outcome skipped, Detail compiler not found]
    ST -->|csTimeout| SK2[Outcome skipped, Detail timeout]
    ST -->|csFail| PF{ExpectErrorText of the source?}
    PF -->|yes, the text is in Log| AC1[Outcome accepted — pitfall, the demanded error appeared]
    PF -->|yes, the text not in Log| WE[Outcome wrong-error, Detail FirstCompilerErrorLine of Log — for Oxygene the first E: line]
    PF -->|no| CE[Outcome compile-error, Detail FirstCompilerErrorLine of Log]
    ST -->|csOk| P2{ExpectErrorText of the source?}
    P2 -->|yes| PO[Outcome compiled-but-should-fail — T06, unchanged]
    P2 -->|no| DF2[default: Outcome skipped, Detail could not start — kept only when the run does not start]
    DF2 --> RU{Dialect is D_OXYGENE?}
    RU -->|yes| RO[RunLimited flatpak run, command dotnet, AEnv.Elements, Bin/Ex.dll — in W, its own work dir]
    RU -->|no| RN[RunLimited ExePath, no args — in W, T06]
    RO --> RST{Ran, TimedOut, Code}
    RN --> RST
    RST -->|Ran and TimedOut| TO2[Outcome timeout, Detail ATimeoutSec]
    RST -->|Ran and Code not 0| RE2[Outcome run-error, Detail the exit code]
    RST -->|not started| KS[Outcome kept skipped, Detail could not start]
    RST -->|Ran| EX{.expected next to the source?}
    EX -->|readable and byte-for-byte equal to RunOut| AC2[Outcome accepted]
    EX -->|readable but different| WO[Outcome wrong-output — no partial match]
    EX -->|not readable| NE[Outcome no-expected]
    AC1 --> CLN{Dialect is D_OXYGENE?}
    WE --> CLN
    CE --> CLN
    PO --> CLN
    SK1 --> CLN
    SK2 --> CLN
    TO2 --> CLN
    RE2 --> CLN
    KS --> CLN
    AC2 --> CLN
    WO --> CLN
    NE --> CLN
    CLN -->|yes| RM2[RemoveTree W — the work dir under OxyRoot is removed, whatever the outcome]
    CLN -->|no| KP[the work dir under AWorkRoot is left as T06 — git-ignored, e.g. work/verify/n]
    RM2 --> AOK2[AOK := IsGoodOutcome of the Outcome — T06: accepted, static-ok, skipped are good]
    KP --> AOK2
    AOK2 --> TSV([Return the TSV line: path, dialect, class, outcome, NormalizeDetail of the detail])
```

Notes on the run step: the program runs **inside its fresh work directory** (run it where the
compiler put it) — for Oxygene, `dotnet Bin/Ex.dll` with `W` as the working directory. The
`skipped` / `could not start` default is the defensive keep-alive for a run that never starts.

## Branch / check coverage (UT and ITa rows)

| Branch / check | Condition in the UTs | UT / ITa case |
|----------------|----------------------|---------------|
| `ReadEnvRec`: `Elements` the app id, `OxyRoot` under `AHome` | `ReadEnvRec('', '', '/h')` | UT-DialectVerify-301 |
| `ChooseCompiler` Oxygene arm: `ExeSearch('flatpak')`, class `compiled-run` | `ChooseCompiler('oxygene', ReadEnvRec('', '', '/h'), Cls)` | UT-DialectVerify-013 |
| `FirstCompilerErrorLine`: the `E: ` prefix, positional and case-sensitive | the six fixture lines, `XE:`, indented, lowercase rejected | UT-DialectVerify-302 |
| `OxygeneStep`: `csOk` with `Program.pas`, `Ex.elements`, `Bin/Ex.dll` in the work dir | the good source into `<OxyRoot>/a`, `ExeSearch('flatpak')` | UT-DialectVerify-303 |
| `OxygeneStep`: `csFail`, `E: ` line in the detail, work dir not in the log | the bad source into `<OxyRoot>/b` | UT-DialectVerify-303 |
| `OxygeneStep`: no flatpak → `csNoCompiler`; 1 ms limit → `csTimeout` | `''` and `/nonexistent/flatpak`; 1 ms | UT-DialectVerify-303 |
| `RemoveTree`: recursive, parent kept | three levels of files under `t` | UT-DialectVerify-304 |
| `VerifyFile` Oxygene: accepted, wrong-output, no-expected, run-error `3`, pitfall accepted, wrong-error `E: … (8)` | the six sources under `work/ut/dv-305/s`, `AWorkRoot` `work/ut/dv-305/w` | UT-DialectVerify-305 |
| `VerifyFile` Oxygene clean-up: no entry under `OxyRoot`, `AWorkRoot` neither created nor deleted | afterwards, after the six runs of the previous row | UT-DialectVerify-305 |
| `VerifyFile` Oxygene `compile-error`: detected dialect and forced dialect | `namespace Demo; type T = class of object end; end.`; `program p; begin end.` with forced `oxygene` | UT-DialectVerify-103, 104 |
| Oxygene record through the dataset CLI: `compiled-run`, `compile-error`, detail `E: Could not find a "Main" method in this project` | the record `namespace N; end.` | UT-DatasetCli-105 |
| Oxygene reply through the eval CLI: `compile-error`, not compiled, not passed | the reply `namespace N;` | UT-EvalCli-105 |
| 11 Oxygene files: outcome as planted, class `compiled-run` | `tests/ita/T22/files`, `--dialect oxygene` | ITa-T22-001 |
| details free of the home and work directory paths | the TSV detail column | ITa-T22-002 |
| no work directories left under `~/.cache/qlora-verify` | after the run | ITa-T22-003 |
| seconds per Oxygene file (build and run) ≤ 5 | ~0.9 s measured | ITa-T22-004 |
| the T19 ITa again with this `src/`: 6 OK, 0 NG; `x-oxygene` rejected with `verify: wrong-output` | `tests/ita/T19-run.sh` with the T22 `src/` | ITa-T22-005, 006 |

## Change history

| Date | Change |
|------|--------|
| 2026-11-25 | Initial program flow |
