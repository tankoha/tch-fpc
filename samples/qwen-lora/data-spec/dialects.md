# Dialects (spec-v1)

Five dialects, each checked with its own compiler. Write only what the
dialect's section allows. A sample tags the features it uses
(`features`); **every feature of a dialect appears in at least 60 of its
1000 samples** (the harness asks for each in turn).

## `objfpc` — Free Pascal 3.2.2, `{$mode objfpc}`

- **Declaration**: the source starts with `program <Name>;` (or `unit
  <Name>;`) followed by `{$mode objfpc}{$H+}` on its own line, before
  `uses`.
- **Libraries**: the Free Pascal 3.2.2 RTL and FCL units `SysUtils`,
  `Classes`, `Math`, `StrUtils`, `DateUtils`, `Types`, `TypInfo`,
  `Variants`, `Contnrs`, `fgl`, `Generics.Collections`,
  `Generics.Defaults`.
- **Features**:
  | Id | Feature |
  |---|---|
  | `objfpc.class` | classes with fields, methods, properties, constructors and destructors; objects freed in `try ... finally` |
  | `objfpc.inheritance` | virtual and override methods, abstract methods, `inherited` |
  | `objfpc.interface` | interfaces (`IInterface` descendants, reference counted) |
  | `objfpc.generic` | `generic` type declarations and `specialize` at the use site |
  | `objfpc.generic-routine` | generic functions called with `specialize F<T>(...)` |
  | `objfpc.fgl` | `fgl` lists and maps (`TFPGList`, `TFPGMap`) via `specialize` |
  | `objfpc.method-pointer` | `of object` types and procedure variables assigned with `@` |
  | `objfpc.exception` | own exception classes, `raise ... CreateFmt`, `on E: T do` |
  | `objfpc.record` | advanced records with methods (`{$modeswitch advancedrecords}`) |
  | `objfpc.set-enum` | enumerations, sets, `case`, `for ... in` over sets and arrays |
  | `objfpc.string` | AnsiString work with `Copy`, `Pos`, `Format`, `StrUtils` routines |
  | `objfpc.dynarray` | dynamic arrays, `SetLength`, open array parameters |
- **Not in this dialect**: inline `var` in a `begin ... end` block,
  `TList<T>` without `specialize`, method pointers without `@`, anonymous
  functions.

## `fpc-delphi` — Free Pascal 3.2.2, `{$mode delphi}`

- **Declaration**: `program <Name>;` (or `unit`) followed by
  `{$mode delphi}` on its own line. No `{$H+}` needed.
- **Libraries**: as `objfpc`.
- **Features**:
  | Id | Feature |
  |---|---|
  | `fpcdelphi.class` | classes, properties, constructors, destructors; `try ... finally` |
  | `fpcdelphi.inheritance` | virtual, override, abstract, `inherited` |
  | `fpcdelphi.interface` | interfaces and reference counting |
  | `fpcdelphi.generic` | generic types written `TBox<T>`, used `TBox<Integer>` (no `generic`/`specialize`) |
  | `fpcdelphi.generic-routine` | generic functions called with an explicit type argument, `MaxOf<Integer>(...)` |
  | `fpcdelphi.collections` | `Generics.Collections`: `TList<T>`, `TDictionary<K, V>`, `TObjectList<T>` |
  | `fpcdelphi.method-pointer` | method pointers assigned without `@` |
  | `fpcdelphi.exception` | own exception classes, `on E: T do`, `raise` |
  | `fpcdelphi.record` | records with methods and operators |
  | `fpcdelphi.set-enum` | enumerations, sets, `case`, `for ... in` |
  | `fpcdelphi.string` | strings with `Copy`, `Pos`, `Format`, `StrUtils` |
  | `fpcdelphi.dynarray` | dynamic arrays and open arrays |
- **Not in this dialect**: inline `var`, `generic`/`specialize`, `@` for
  method pointers, anonymous methods, `System.`-prefixed unit names.

## `unleashed` — Unleashed Pascal, `{$mode unleashed}`

- **Declaration**: `program <Name>;` (or `unit`) followed by
  `{$mode unleashed}` on its own line.
- **Libraries**: as `objfpc`, plus `cthreads` (first in `uses`) when the
  sample uses `async`, `for parallel` or `lock`.
- **Features**:
  | Id | Feature |
  |---|---|
  | `unleashed.inline-var` | inline variables `var X := ...;` and `for var I := ...` |
  | `unleashed.generic` | generic types and routines written `<T>`, explicit type arguments on calls |
  | `unleashed.method-pointer` | method pointers and procedure variables assigned with `@` |
  | `unleashed.if-expr` | `if`, `case` and `try` used as expressions (one value per branch) |
  | `unleashed.match` | `match` expressions with a `_` branch |
  | `unleashed.interpolation` | string interpolation `$'...{expr}...'` |
  | `unleashed.autofree` | `autofree` and `defer X.Free;` for cleanup |
  | `unleashed.async` | `async` and `await` (with `cthreads`), the result read after `await` |
  | `unleashed.parallel` | `for parallel var I := ...` with shared state guarded by `lock` or Interlocked |
  | `unleashed.class` | classes, inheritance, properties, interfaces |
  | `unleashed.exception` | exceptions, own exception classes |
  | `unleashed.collections` | `Generics.Collections` with `<T>` |
- **Rules**: inline variables are initialised with `:=`, `var`-section
  initialisers with `=`; expression branches hold one value each (no
  `raise`, `exit` or `begin ... end` inside them); parallel and async
  results are printed in a fixed order.
- **Not in this dialect**: `generic`/`specialize`.

## `delphi` — Delphi (checked as an approximation with FPC)

Delphi samples are compiled by Free Pascal in Delphi-Unicode mode with
anonymous functions, function references and inline variables, after the
`System.`-prefixed unit names are mapped to FPC's; so they must stay
within what both compilers accept.
- **Declaration**: `program <Name>;` followed by `{$APPTYPE CONSOLE}` on
  its own line; no `{$mode ...}` directive. (A unit: `unit <Name>;`
  without the directive.)
- **Libraries**: `System.SysUtils`, `System.Classes`, `System.Math`,
  `System.StrUtils`, `System.DateUtils`, `System.Types`,
  `System.Variants`, `System.TypInfo`, `System.Generics.Collections`,
  `System.Generics.Defaults`. Nothing from VCL or FMX.
- **Features**:
  | Id | Feature |
  |---|---|
  | `delphi.class` | classes, properties, constructors, destructors; `try ... finally` |
  | `delphi.inheritance` | virtual, override, abstract, `inherited` |
  | `delphi.interface` | interfaces and reference counting |
  | `delphi.generic` | generic types `<T>` with constraints `<T: class, constructor>` |
  | `delphi.generic-method` | generic methods as `class function F<T>` of a class or record (no global generic routines) |
  | `delphi.collections` | `TList<T>`, `TDictionary<K, V>`, `TObjectList<T>` |
  | `delphi.inline-var` | `var X := ...;` and `for var I := ...` |
  | `delphi.anonymous` | anonymous methods assigned to `reference to` types declared in the sample (`TProc` / `TFunc` of `System.SysUtils` are not available in the check) |
  | `delphi.string-helper` | string helpers: `S.Length`, `S.ToUpper`, `S.StartsWith`, `S.Split` |
  | `delphi.exception` | own exception classes, `on E: T do` |
  | `delphi.record` | records with methods and operators |
  | `delphi.set-enum` | enumerations, sets, `case`, `for ... in` |
- **Not in this dialect**: managed-record operators (`Initialize` /
  `Finalize`), FPC units (`fgl`, `Contnrs`), `{$mode}` directives,
  global generic routines, `TProc` / `TFunc` (declare the `reference to`
  type yourself).
- **Rules of the check**: a list of numbers in `for X in [...]` is a set
  of bytes there (use a typed constant array); sorting strings with the
  default comparer fails (give `TComparer<string>.Construct` with
  `CompareStr`); nested generic types are declared as aliases
  (`TWordList = TList<TWordCount>`), not written inline.

## `oxygene` — RemObjects Oxygene (Elements 14), .NET Core

- **Declaration**: the file starts with `namespace <Name>;` and ends
  with `end.`; one file per sample; the entry point is `class method
  Main(args: array of String): Int32;` of a class `Program`.
- **Libraries**: the .NET base library through `uses`: `System`,
  `System.Collections.Generic`, `System.Linq`, `System.Text`. No file,
  network or GUI namespaces.
- **Features**:
  | Id | Feature |
  |---|---|
  | `oxygene.class` | classes with `method`, nameless `constructor`, `new`, properties (no `Free`) |
  | `oxygene.inheritance` | virtual and override methods, abstract classes, `inherited` |
  | `oxygene.interface` | interfaces and their implementation |
  | `oxygene.generic` | generic types and methods `<T>` with `where` constraints |
  | `oxygene.collections` | `List<T>`, `Dictionary<K, V>`, `for each ... in ... index i` |
  | `oxygene.lambda` | lambdas `x -> expr` and anonymous methods |
  | `oxygene.linq` | LINQ query operators on sequences (`Where`, `Select`, `OrderBy`, `Sum`) |
  | `oxygene.interpolation` | string interpolation `$'...{expr}...'`; 0-based immutable strings |
  | `oxygene.exception` | own exception classes, `raise new ...`, `on E: T do` |
  | `oxygene.event` | events with `block` types, subscribed with `+=` |
  | `oxygene.record` | records and enumerations, `case` |
  | `oxygene.loop` | `for ... step`, `downto`, `repeat`, `while`, `loop` with `break` |
- **Rules**: results via `result` or `exit value`; `var`/`out` written at
  the call site too; comments that contain braces use `(* ... *)`;
  compare unconstrained generic values with `Equals`, not `=`.
- **Not in this dialect**: `unit`, `procedure`/`function` keywords,
  `Free`, `destructor`, `initialization`/`finalization`, `with`, `asm`,
  `reference to`.
