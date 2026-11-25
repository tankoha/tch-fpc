# T22 External Design: Oxygene Build and Run in the Filter

## Purpose

T22 adds Oxygene (RemObjects Elements) as a supported Pascal dialect in the
`qlora_verify` filter. Oxygene sources are compiled with EBuild inside the
Elements flatpak and run with the flatpak's dotnet, producing the class
`compiled-run` like the other compiled dialects (Delphi, FPC, FPC Unleashed).

## Build and Run

The build follows the `pascal-oxygene` skill's `check-examples.sh`:

1. The source file is copied as `Program.pas` into a work directory.
2. A project file `Ex.elements` (.NET Core, Echoes) is written next to it.
3. EBuild is invoked inside the Elements flatpak:
   `flatpak run --command=/app/dotnet/dotnet com.remobjects.earth /app/earth/EBuild.dll Ex.elements`
4. A build is successful when EBuild exits 0 **and** `Bin/Ex.dll` exists.
5. On success, the DLL is run with:
   `flatpak run --command=/app/dotnet/dotnet com.remobjects.earth Bin/Ex.dll`

The flatpak is discovered with `ExeSearch('flatpak')`. Without it, the outcome
is `skipped` with detail `compiler not found` (same as for a missing FPC).

## Work Directory Location and Removal

The work directory for Oxygene is under the home directory:
`<HOME>/.cache/qlora-verify/<n>` (`TEnvRec.OxyRoot`).

**Why under the home directory:** The flatpak sees the host's files but has a
`/tmp` of its own. A work directory under `/tmp` would be invisible to the
flatpak.

**Removal:** The work directory is deleted after verification (`RemoveTree`),
whatever the outcome. This ensures no accumulation of temporary files even on
build failures or timeouts. The normal `AWorkRoot` is not used for Oxygene.

## One File per Oxygene Record

Only the source itself is compiled; units next to it are not. An Oxygene sample
that has a class and its test program keeps both in one namespace file, as the
data specification requires.

## Outcomes and Details

| Outcome | Detail | Condition |
|---------|--------|-----------|
| `compile-error` | EBuild's first `E: ` line | Build fails (exit code ≠ 0 or no `Bin/Ex.dll`) |
| `wrong-error` | EBuild's first `E: ` line | Build fails and `expect-error:` marker text not in log |
| `run-error` | Exit code | Run exits with non-zero code |
| `timeout` | Time limit seconds | The run exceeds the time limit |
| `wrong-output` | (none) | Output differs from expected file |
| `no-expected` | (none) | No expected file found |
| `accepted` | (none) | Run succeeds and output matches expected; or the build fails with the text of the source's `expect-error:` marker in the log |
| `skipped` | `compiler not found` | Flatpak not installed |
| `skipped` | `timeout` | Build exceeds time limit |

EBuild's error lines start with `E: ` (e.g., `E: Unknown identifier "Foo" [Program.pas (7)]`).
The work directory is removed from the log before constructing the detail, so
the detail names only `Program.pas`, never a path under the home directory.
