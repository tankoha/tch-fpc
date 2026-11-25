# Detailed design — Oxygene build and run in the filter (T22)

Based on the external design `docs/external-design/T22-external.md`, the system flow
`docs/system-flow/T22-system.md`, and the facts `tasks/T22-facts.md` (final, checked by the
harness owner). The functions T22 changes are those designed in
`docs/detailed-design/T06-detailed.md` (the unit `DialectVerify`). This document restates and
organises the facts; it adds no new requirements.

T22 replaces the static Oxygene check (T06: `static-ok` / `static-reject`, nothing compiled)
with a build and a run: the Oxygene source is built with EBuild inside the RemObjects Elements
flatpak and executed with the flatpak's `dotnet`, so Oxygene joins the compiled-and-run
dialects; its class is `compiled-run`, like Delphi, FPC and FPC Unleashed. One unit,
`DialectVerify` (T06), changes; `qlora_verify`, `qlora_dataset` (T19) and `qlora_eval` (T20)
use it unchanged (the users are `docs/system-flow/T22-system.md`).

The implementation below is final, from the facts ("write it exactly so"); the Pascal snippets
are copied verbatim from the facts.

## 1. Module structure

| Source file | T22 change |
|-------------|-----------|
| `src/DialectVerify.pas` | `TEnvRec` gains the fields `Elements`, `OxyRoot`; `ReadEnvRec` sets them (two lines at the end); the Oxygene branch of `ChooseCompiler` becomes the `flatpak` lookup with the class `compiled-run`; `FirstCompilerErrorLine` also takes EBuild's `E: ` lines; `OxygeneStep` and `RemoveTree` get their bodies (replacing the skeleton); `VerifyFile` gets a new whole body (the static Oxygene branch goes, the Oxygene work-dir choice, build, run and clean-up come in). The constant `K_OXY_PROJECT` is already in the skeleton and is kept as-is |
| `src/qlora_verify.pas`, the `T19` / `T20` CLI units | unchanged |

`VS_STATIC_ONLY`, `OUT_STATIC_OK` and `OUT_STATIC_REJECT` stay as constants (other units name
them) but are no longer produced by `VerifyFile`.

## 2. `TEnvRec` and `ReadEnvRec`

The record gains two fields (verbatim from the facts):

```pascal
  TEnvRec = record
    Fpc, FpcUnleashed: string;
    { T22: the Elements flatpak (app id) and the root of the Oxygene work
      directories (under the home directory) }
    Elements, OxyRoot: string;
  end;
```

`ReadEnvRec` (T06 §2.1: `Fpc` = `AFpc` or `'fpc'`, `FpcUnleashed` = `AFpcUnleashed` or
`AHome + '/unleashed/fpc/bin/fpc'`, the environment values passed by the caller — the program
`qlora_verify` is the only environment reader) sets them at its end, with these two lines
(verbatim):

```pascal
  Result.Elements := 'com.remobjects.earth';
  Result.OxyRoot := AHome + '/.cache/qlora-verify';
```

- `Elements` is the Elements flatpak app id; it is a fixed constant, not an environment value.
- `OxyRoot` is the root of the Oxygene work directories, under the home directory:
  `<HOME>/.cache/qlora-verify`. **Why under the home directory:** the flatpak sees the host's
  files but has a `/tmp` of its own — a work directory under `/tmp` would be invisible to the
  flatpak. The normal `AWorkRoot` parameter is not used for Oxygene (§8).
- `UT-DialectVerify-301`: `ReadEnvRec('', '', '/h')` → `Elements` = `com.remobjects.earth`,
  `OxyRoot` = `/h/.cache/qlora-verify`.

## 3. `ChooseCompiler` — the Oxygene branch

The Oxygene arm of the dialect dispatch becomes (verbatim):

```pascal
  else if D = D_OXYGENE then
  begin
    Result := ExeSearch('flatpak');  { T22: EBuild in the Elements flatpak }
    AClass := VS_COMPILED_RUN;
  end
```

- The returned "compiler path" is the `flatpak` executable (found in `PATH`, the way FPC is
  found by `IsPathUsable` / `ExeSearch`); the real compiler, EBuild, runs inside the app
  `AEnv.Elements` later (§6, §8).
- The class is now `compiled-run` (T06 gave `static-only`), like the other compiled dialects.
- Without `flatpak`, `ExeSearch('flatpak')` is `''`; `OxygeneStep` then returns
  `csNoCompiler` and `VerifyFile` classifies `skipped` / `compiler not found` — the same shape
  as a missing FPC, not a failure.
- `UT-DialectVerify-013` (changed): `ChooseCompiler('oxygene', ReadEnvRec('', '', '/h'), Cls)`
  → `ExeSearch('flatpak')`, with `Cls` = `compiled-run`.

## 4. `FirstCompilerErrorLine` — the `E: ` prefix

The acceptance condition of the first matching line gains a third alternative (verbatim):

```pascal
    if (Pos('error:', LowLine) > 0) or (Pos('fatal:', LowLine) > 0) or
       (Copy(Line, 1, 3) = 'E: ') then  { T22: EBuild's error lines }
      Result := Line;
```

- FPC's `error:` / `fatal:` lines (lowercased as in T06) are unchanged; EBuild's error lines are
  added: any line whose first three characters are exactly `E`, `:`, space.
- EBuild's error lines start with `E: `, e.g. `E: Unknown identifier "Foo" [Program.pas (7)]`.
- The prefix test is positional and case-sensitive: `XE: not this`, an indented ` E: ...` and
  the lowercase `e: lower` do not match (`UT-DialectVerify-302`).
- No new message: the line is used as the `<detail>` cell (normalized as in T06 §2.8) of
  `compile-error` / `wrong-error`, after the work directory has been removed from the log (§6).

## 5. `K_OXY_PROJECT` — the project file

`K_OXY_PROJECT` (already in the `DialectVerify` skeleton, kept as-is) is the text of the
project file `Ex.elements` that `OxygeneStep` writes next to the copied source. It defines a
.NET Core (Echoes) project named `Ex` for EBuild: building it in the work directory compiles
the single `Program.pas` sitting next to it into `Bin/Ex.dll` — which is exactly what the
result check of §6 tests for.

- Only the source itself is compiled (units next to it are not), so there is **one file per
  Oxygene record**: an Oxygene sample that has a class and its test program keeps both in one
  namespace file (the data specification says so).
- The literal text of the project file is this constant; `OxygeneStep` writes it unchanged to
  `<workdir>/Ex.elements` (§6) and the build command names it: `EBuild.dll Ex.elements`.

## 6. `OxygeneStep`

Signature (already in the skeleton; the body is new, replacing the skeleton's):

```pascal
function OxygeneStep(AFlatpak, AApp, ASrcPath, AOutDir: string; ATimeoutMs: Integer;
  out ALog: string): TCompileState;
```

| Argument | Meaning (from `VerifyFile`, §8) |
|----------|--------------------------------|
| `AFlatpak` | the `flatpak` executable returned by `ChooseCompiler` (§3) |
| `AApp` | the app id `AEnv.Elements` (`com.remobjects.earth`) |
| `ASrcPath` | the Oxygene source under verification |
| `AOutDir` | the fresh work directory `<OxyRoot>/<n>` (under the home directory) |
| `ATimeoutMs` | the time limit `ATimeoutSec * 1000` |
| `ALog` | out: the build log with the work directory removed from it |

**A build is good only when EBuild exits 0 AND `Bin/Ex.dll` exists** — both must hold (the
facts' decision). The work directory is created here; its removal is `RemoveTree` in
`VerifyFile` (§7, §8).

The body (verbatim from the facts):

```pascal
{ 2.13b (T22) Oxygene: copy the source as Program.pas into AOutDir (under
  the home directory: the flatpak's /tmp is its own), write the project
  Ex.elements (.NET Core, Echoes) and build it with EBuild inside the
  Elements flatpak. csOk when EBuild exits 0 and Bin/Ex.dll exists. }
function OxygeneStep(AFlatpak, AApp, ASrcPath, AOutDir: string; ATimeoutMs: Integer;
  out ALog: string): TCompileState;
var
  Text, Out: string;
  Code: Integer;
  TimedOut: Boolean;
  L: TStringList;
begin
  ALog := '';
  if (AFlatpak = '') or (not IsPathUsable(AFlatpak)) then
    Exit(csNoCompiler);
  ForceDirectories(AOutDir);
  ReadTextFile(ASrcPath, Text);
  L := TStringList.Create;
  try
    L.Text := Text;
    L.SaveToFile(AOutDir + '/Program.pas');
    L.Text := K_OXY_PROJECT;
    L.SaveToFile(AOutDir + '/Ex.elements');
  finally
    L.Free;
  end;
  RunLimited(AFlatpak, ['run', '--command=/app/dotnet/dotnet', AApp, '/app/earth/EBuild.dll',
    'Ex.elements'], AOutDir, ATimeoutMs, Out, Code, TimedOut);
  ALog := StringReplace(Out, AOutDir + '/', '', [rfReplaceAll]);  { no home path in a detail }
  if TimedOut then
    Exit(csTimeout);
  if (Code <> 0) or not FileExists(AOutDir + '/Bin/Ex.dll') then
    Exit(csFail);
  Result := csOk;
end;
```

Steps, in code order:

1. `ALog := ''`.
2. `AFlatpak = ''` or not usable → `csNoCompiler` (no flatpak → `skipped` /
   `compiler not found`, §3).
3. `ForceDirectories(AOutDir)` — the work directory under `OxyRoot`.
4. Copy the source into the work directory as `Program.pas`, then write `K_OXY_PROJECT` as
   `Ex.elements` next to it (one `TStringList`, freed in `try..finally`).
5. Build inside the flatpak, work directory `AOutDir` (`RunLimited` as in T06 §2.9):
   `flatpak run --command=/app/dotnet/dotnet <app id> /app/earth/EBuild.dll Ex.elements` — the
   `pascal-oxygene` skill's build line (the facts' decision).
6. `ALog := StringReplace(Out, AOutDir + '/', '', [rfReplaceAll])` — the work directory is
   removed from the log, so a detail names `Program.pas` and never a path under the home
   directory (ITa-T22-002; the `UT-303` "directory not in Log" check).
7. `TimedOut` → `csTimeout` (a build over the time limit → `skipped` / `timeout`).
8. `Code <> 0` **or** `Bin/Ex.dll` missing → `csFail` (→ `compile-error` / `wrong-error`, §9).
9. Otherwise `csOk`.

`UT-DialectVerify-303`: the good source under `<OxyRoot>/a` → `csOk` with `Program.pas`,
`Ex.elements` and `Bin/Ex.dll` in `a`; the bad one under `/b` → `csFail` with
`FirstCompilerErrorLine(Log)` = `E: Unknown identifier "Foo" [Program.pas (8)]` and the
directory not in `Log`; a `''` flatpak and `/nonexistent/flatpak` → `csNoCompiler`; the good
source with a 1 ms time limit → `csTimeout`.

## 7. `RemoveTree`

Procedure declaration (already in the skeleton; the body is new, replacing the skeleton's):

```pascal
{ 2.13c (T22) Deletes ADir and everything below it. }
procedure RemoveTree(const ADir: string);
var
  R: TSearchRec;
begin
  if FindFirst(ADir + '/*', faAnyFile or faDirectory, R) = 0 then
  begin
    repeat
      if (R.Name <> '.') and (R.Name <> '..') then
        if (R.Attr and faDirectory) <> 0 then
          RemoveTree(ADir + '/' + R.Name)
        else
          DeleteFile(ADir + '/' + R.Name);
    until FindNext(R) <> 0;
    FindClose(R);
  end;
  RemoveDir(ADir);
end;
```

- Recursive, bottom-up: subdirectories are deleted first (recursion), files with
  `DeleteFile`, and `ADir` itself last with `RemoveDir` — so the whole tree, however nested, is
  gone and nothing below `ADir`'s parent is touched.
- Used for exactly one purpose: the Oxygene work directory, called from `VerifyFile` whatever
  the outcome (§8) — no accumulation of temporary files, not on build failures or timeouts.
- `UT-DialectVerify-304`: with files `t/a.txt`, `t/s/b.txt`, `t/s/u/c.txt` (three levels),
  `RemoveTree('.../t')` → `t` is gone and the parent directory is still there.

## 8. `VerifyFile` — step by step, both paths

`function VerifyFile(ASrcName, AForcedDialect: string; ATimeoutSec: Integer; const AEnv: TEnvRec;
AWorkRoot: string; out AOK: Boolean): string` — the signature, the TSV line and the exit-code
rule are unchanged (T06 §2.14, §2.18); the body is replaced as a whole. The Oxygene static
branch (T06 step 2) goes; the Oxygene work directory, build, run and clean-up come in.
`qlora_verify`, `qlora_dataset` and `qlora_eval` call it unchanged.

The whole body (verbatim from the facts):

```pascal
function VerifyFile(ASrcName, AForcedDialect: string; ATimeoutSec: Integer;
  const AEnv: TEnvRec; AWorkRoot: string; out AOK: Boolean): string;
var
  Dialect, Comp, CClass, Text, Pitfall, Log, ExePath, RunOut, ExpText: string;
  Code: Integer;
  TimedOut, Ran: Boolean;
  State: TCompileState;
  W, Outcome, Detail: string;
begin
  { the source must exist (CLI checks the path); a missing file flows
    through as empty text and is classified, never raised }
  ReadTextFile(ASrcName, Text);
  Log := '';
  Dialect := LowerCase(AForcedDialect);
  if Dialect = '' then
    Dialect := DetectDialect(Text, ASrcName);
  CClass := '';
  Comp := ChooseCompiler(Dialect, AEnv, CClass);
  { fresh out dir <workroot>/<n>, n unique per call (salt + per-process
    sequence), so a rerun never collides with a stale directory }
  GVerifySeq := GVerifySeq + 1;
  if Dialect = D_OXYGENE then
  begin
    W := AEnv.OxyRoot + '/' + IntToStr(GVerifyInit + GVerifySeq);
    State := OxygeneStep(Comp, AEnv.Elements, ASrcName, W, ATimeoutSec * 1000, Log);
  end
  else
  begin
    W := AWorkRoot + '/' + IntToStr(GVerifyInit + GVerifySeq);
    State := CompileStep(Comp, ASrcName, W, ATimeoutSec * 1000,
      (Dialect = D_DELPHI), Log, ExePath);
  end;
  if (State = csNoCompiler) or (State = csStartFail) then
  begin
    Outcome := OUT_SKIPPED;
    Detail := 'compiler not found';
  end
  else if State = csTimeout then
  begin
    Outcome := OUT_SKIPPED;
    Detail := 'timeout';
  end
  else if State = csFail then
  begin
    Pitfall := ExtractExpectErrorText(Text);
    if (Pitfall <> '') and (Pos(Pitfall, Log) > 0) then
      Outcome := OUT_ACCEPTED
    else if Pitfall <> '' then
    begin
      Outcome := OUT_WRONG_ERR;
      Detail := FirstCompilerErrorLine(Log);
    end
    else
    begin
      Outcome := OUT_COMPILE_ERR;
      Detail := FirstCompilerErrorLine(Log);
    end;
  end
  else
  begin
    Pitfall := ExtractExpectErrorText(Text);
    if Pitfall <> '' then
      Outcome := OUT_PIT_OK  { compiled but the source demands a compile error }
    else
    begin
      Outcome := OUT_SKIPPED;
      Detail := 'could not start';
      { T06c fix: the program runs INSIDE its fresh out dir <out> (the facts:
        run it where the compiler put it). RunLimited pre-resolves the relative
        dir to absolute before setting CurrentDirectory, so the absolute-shebang
        exec path below stays valid. Fixing the inherited-cwd default meant
        test 106's fixture now logs to an absolute path (see its notes). }
      if Dialect = D_OXYGENE then
        Ran := RunLimited(Comp, ['run', '--command=/app/dotnet/dotnet', AEnv.Elements,
          'Bin/Ex.dll'], W, ATimeoutSec * 1000, RunOut, Code, TimedOut)
      else
        Ran := RunLimited(ExePath, [], W, ATimeoutSec * 1000, RunOut, Code,
          TimedOut);
      if Ran and TimedOut then
      begin
        Outcome := OUT_TIMEOUT;
        Detail := IntToStr(ATimeoutSec);
      end
      else if Ran and (Code <> 0) then
      begin
        Outcome := OUT_RUN_ERR;
        Detail := IntToStr(Code);
      end
      else if Ran then
      begin
        Detail := '';
        if ReadTextFile(ExpectedPathFor(ASrcName), ExpText) and
           (ExpText = RunOut) then
          Outcome := OUT_ACCEPTED
        else if ReadTextFile(ExpectedPathFor(ASrcName), ExpText) then
          Outcome := OUT_WRONG_OUT
        else
          Outcome := OUT_NO_EXPECTED;
      end;
    end;
  end;
  if Dialect = D_OXYGENE then
    RemoveTree(W);
  AOK := IsGoodOutcome(Outcome);
  Result := ASrcName + #9 + Dialect + #9 + CClass + #9 + Outcome + #9 +
    NormalizeDetail(Detail);
end;
```

Step by step, in code order:

1. `ReadTextFile(ASrcName, Text)` — a missing file flows through as empty text and is
   classified, never raised (the CLI checks the path; T06).
2. `Dialect := LowerCase(AForcedDialect)`; when empty, `DetectDialect(Text, ASrcName)` —
   T06 §2.2, unchanged (`objfpc` / `fpc-delphi` / `unleashed` / `delphi` / `oxygene`).
3. `CClass := ''`; `Comp := ChooseCompiler(Dialect, AEnv, CClass)` — for Oxygene the `flatpak`
   path and the class `compiled-run` (§3).
4. `GVerifySeq := GVerifySeq + 1` — the number `n` is unique per call (salt + per-process
   sequence, T06), so a rerun never collides with a stale directory.
5. **Work directory and compile (this is where the two paths split):**
   - **Oxygene:** `W := AEnv.OxyRoot + '/' + IntToStr(GVerifyInit + GVerifySeq)` and
     `State := OxygeneStep(Comp, AEnv.Elements, ASrcName, W, ATimeoutSec * 1000, Log)` —
     work directory under the home directory (§2), build inside the Elements flatpak (§6).
     `AWorkRoot` is **not used** for Oxygene: it is neither created nor deleted here.
   - **Non-Oxygene:** `W := AWorkRoot + '/' + IntToStr(GVerifyInit + GVerifySeq)` and
     `State := CompileStep(Comp, ASrcName, W, ATimeoutSec * 1000, (Dialect = D_DELPHI), Log,
     ExePath)` — T06 §2.13 (flags, delphi copy, `csOk` = exit 0 and the executable exists),
     unchanged.
6. **Build-state classification (common to both paths, in code order):**
   - `csNoCompiler` or `csStartFail` → `skipped`, detail `compiler not found`;
   - `csTimeout` → `skipped`, detail `timeout`;
   - `csFail`, with `Pitfall := ExtractExpectErrorText(Text)`:
     - pitfall text present in `Log` → `accepted`;
     - pitfall text absent from `Log` → `wrong-error`, detail `FirstCompilerErrorLine(Log)`;
     - no pitfall → `compile-error`, detail `FirstCompilerErrorLine(Log)`.
     For Oxygene that detail is EBuild's first `E: ` line, work directory already removed
     (§4, §6 step 6).
   - `csOk` (the remaining state):
     - with a pitfall → `compiled-but-should-fail` (`OUT_PIT_OK`: it compiled, but the source
       demands a compile error — T06 R11, unchanged);
     - without one → the run step (7).
7. **Run (common to both paths, branched on the dialect).** The outcome defaults to
   `skipped` / `could not start` and is kept only if the run does not start:
   - **Oxygene:** `RunLimited(Comp, ['run', '--command=/app/dotnet/dotnet', AEnv.Elements,
     'Bin/Ex.dll'], W, ATimeoutSec * 1000, ...)` — the built `Bin/Ex.dll` is run with the
     flatpak's `dotnet`, **inside its fresh work directory `W`** (run it where the compiler
     put it; `RunLimited` pre-resolves the relative directory to absolute, T06 §2.9).
   - **Non-Oxygene:** `RunLimited(ExePath, [], W, ATimeoutSec * 1000, ...)` — the compiled
     executable in `<out>`, T06.
   - Then, in code order: `Ran and TimedOut` → `timeout`, detail `IntToStr(ATimeoutSec)`;
     `Ran and Code <> 0` → `run-error`, detail `IntToStr(Code)`; `Ran` → compare with the
     expected file (T06, byte-for-byte): `ReadTextFile(ExpectedPathFor(ASrcName), ExpText) and
     ExpText = RunOut` → `accepted`; the expected file exists but differs → `wrong-output`;
     no expected file → `no-expected`.
8. **Clean-up (Oxygene only):** `if Dialect = D_OXYGENE then RemoveTree(W)` — after *every*
   outcome, the Oxygene work directory under `OxyRoot` is deleted (§7); the non-Oxygene work
   directory under `AWorkRoot` (e.g. `work/verify/<n>`) is left as T06 (git-ignored).
9. `AOK := IsGoodOutcome(Outcome)` (T06 §2.18: `accepted` / `static-ok` / `skipped` are good)
   and the TSV line `ASrcName<tab>Dialect<tab>CClass<tab>Outcome<tab>NormalizeDetail(Detail)`
   (T06 §2.16, detail ≤ 200 chars, tabs → blanks).

## 9. Outcomes and details

No new outcome and no new message (the facts' "Checks and messages"). The vocabulary is T06's;
for Oxygene each outcome is reached as follows:

| Outcome | Detail | How Oxygene reaches it (T22) |
|---------|--------|------------------------------|
| `skipped` | `compiler not found` | no `flatpak` on the system: `OxygeneStep` `csNoCompiler` (§3, §6 step 2) |
| `skipped` | `timeout` | the build over the time limit: `OxygeneStep` `csTimeout` (§6 step 7) |
| `skipped` | `could not start` | `dotnet` run does not start (defensive; the flatpak was just used for the build) |
| `compile-error` | EBuild's first `E: ` line (work directory removed from it) | `csFail`, no `expect-error:` marker in the source |
| `wrong-error` | EBuild's first `E: ` line | `csFail`, `expect-error:` marker's text not in the log |
| `accepted` | (empty) | marker's text is in the log (pitfall accepted, §6), **or** the run exits 0 and the output is byte-for-byte the `.expected` file |
| `compiled-but-should-fail` | (empty) | `csOk`, but the source carries an `expect-error:` marker |
| `run-error` | the exit code | the run exits non-zero |
| `timeout` | the time limit in seconds | the run over the time limit |
| `wrong-output` | (empty) | the output differs from the `.expected` file (no partial match, T06) |
| `no-expected` | (empty) | no `.expected` file next to the source |

External-design rule → where it is realized here:

| Rule (`T22-external.md`) | Where |
|---|---|
| copy as `Program.pas`, write `Ex.elements` (.NET Core, Echoes), the EBuild line, run line, `Bin/Ex.dll` | §5, §6 steps 4–5, §8 step 7 |
| a build is good iff EBuild exits 0 **and** `Bin/Ex.dll` exists | §6 step 8 |
| work directory under the home directory, why; removed after whatever outcome; `AWorkRoot` not used | §2, §8 steps 5 and 8 |
| one file per Oxygene record | §5 |
| flatpak via `ExeSearch('flatpak')`; without it `skipped` / `compiler not found` | §3, §6 step 2 |
| `E: ` lines as details; work directory removed from the log | §4, §6 step 6, this section |

## 10. Notes (from the facts)

- **UT** (T22b, the harness owner's fixture and helpers, untouched by this task): every row of
  `DialectVerify_test` passes with `-Cr` (108 rows: the earlier ones, rows `013`, `103`, `104`
  changed as in the facts, rows `301` to `305`); `DialectVerify.pas` then has 100 % line
  coverage (551 lines); the mutants named in the facts are caught (EBuild's lines not taken;
  Oxygene left without a compiler; a failed build taken as good; the work directory left in the
  log; the work directory not removed; `RemoveTree` not recursive; the build's time limit
  ignored; another app id; another `OxyRoot`).
- **ITa** (T22c): `ITa-T22-001` to `006` (outcomes and class of 11 Oxygene files; details free
  of home/work-directory paths; no work directories left under `~/.cache/qlora-verify`; ≤ 5 s
  per Oxygene file; the T19 ITa again with 0 NG cases; `x-oxygene` rejected with
  `verify: wrong-output`).
- **Stages** (the facts): stage 1 (T22b) is the "Implementation" above with rows
  `UT-DialectVerify-013`, `103`, `104` (changed) and `301` to `305`; the ITa is
  `ITa-T22-001` to `006`. About 0.9 s per Oxygene file (build and run), measured by the
  harness owner.
- `VS_STATIC_ONLY`, `OUT_STATIC_OK` and `OUT_STATIC_REJECT` stay as constants (other units name
  them) but are no longer produced.

## 11. Change history

| Date | Change | Reason |
|------|--------|--------|
| 2026-11-25 | Initial creation | T22a, delegation 1 of 2: the T22 detailed design (no source code, no tests) |
