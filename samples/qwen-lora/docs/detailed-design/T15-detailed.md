# Detailed Design — Unit `TrainData` (T15)

Source of truth: `../external-design/T15-external.md`,
`../system-flow/T15-system.md` and the T15 facts
(`tasks/T15-facts.md`). This document restates and organises them; it adds
no new requirements.

## 1. Module composition

- Source file: `src/TrainData.pas`, `{$mode objfpc}{$H+}`.
- Units used: `QwenTokenizer` (T05: `TQwenTokenizer`, `TTokenArray`),
  `QwenChat` (T07: `TChatMessages`, `TChatOptions`, `ChatMsg`, `crUser`,
  `crAssistant`, `DefaultChatOptions`, `EncodeTraining`), `fpjson`
  (`GetJSON`, `TJSONObject`, `TJSONData`, `jtString`, `EJSONParser`),
  `Tch` (`TTensor`, `TTensor.FromArray`, `.view`), `System` (`Max`,
  `Min`, `QWord`, `Exception.Create`/`CreateFmt`), `System.SysUtils`
  (`FileExists`), `System.Classes` (`TStringList`).
- Declared (interface section):
  - `ETrainData = class(Exception)` — the unit's error class.
  - `TIntArray` — a dynamic array of `Integer` (4.1).
  - `TTrainSample` — one training sample: id, dialect, tokens and the
    loss span (4.1).
  - `TBatch` (a batch as indices into the set's samples) and `TBatches`
    (all the batches of one epoch).
  - `TTrainSet` — the loaded data set: private fields `FSamples`,
    `FSkippedClass`, `FSkippedLong`; private method `GetSample`; public
    `Create`, `Count`, `MaxLength`, the default property `Samples[I]` and
    the properties `SkippedClass`, `SkippedLong`.
  - Functions: `EncodeSample`, `EpochOrder`, `EpochBatches`,
    `SkipBatches`, `BatchTensors`.
- Local routines (implementation section only): `SplitMix64`, `StateOf`
  (both inside a `{$push}{$Q-}{$R-}` … `{$pop}` block), `Shuffle`,
  `HasWideEscape`, `GetStr`.
- Stages (T15 facts, "Stages and case IDs"): T15b1 writes `SplitMix64`,
  `StateOf`, `Shuffle` (used in stage 2), `EncodeSample`,
  `HasWideEscape`, `GetStr`, `TTrainSet` (all of it); T15b2 writes
  `EpochOrder`, `EpochBatches`, `SkipBatches`, `BatchTensors`. The
  interface section is complete and final for both stages.

## 2. Function specifications

| Function | Arguments | Returns | Raises | Preconditions |
|---|---|---|---|---|
| `SplitMix64` (local) | `var S: QWord` | the next SplitMix64 value (`QWord`); `S` advanced | — | — |
| `StateOf` (local) | `ASeed, AEpoch, APool: Int64` | the shuffle state `QWord` | — | — |
| `Shuffle` (local) | `var A: TIntArray; AState: QWord` | — (`A` shuffled in place, Fisher–Yates) | — | — |
| `EncodeSample` | `Tok: TQwenTokenizer; const APrompt, AAnswer: RawByteString; out LossFrom, LossTo: Integer` | the tokens of the rendered conversation (`TTokenArray`); the loss span in `LossFrom`/`LossTo` | the `QwenChat` error of `EncodeTraining` — none for these records (a user and an assistant message are always present) | the tokenizer alive |
| `HasWideEscape` (local) | `const L: RawByteString` | `True` when the line holds a `\u` escape of a character above U+007F | — | — |
| `GetStr` (local) | `O: TJSONObject; const AKey: string; ALine: Integer` | the string of `AKey` (`RawByteString`) | `ETrainData` (4.10 #5) | `O` alive |
| `TTrainSet.Create` | `Tok: TQwenTokenizer; const AFile: string; const AClasses: array of string; AMaxLen: Integer` | the new set | `ETrainData` (4.10 #1–#6), the `EncodeSample` raises (none for these records) | — |
| `TTrainSet.Count` | — | the number of admitted samples | — | created |
| `TTrainSet.MaxLength` | — | the length of the longest sample; 0 for none | — | created |
| `TTrainSet.GetSample` (private; the property `Samples[I]`) | `I: Integer` | the sample `FSamples[I]` | `ETrainData` (4.10 #7) | created; `0 <= I < Count` |
| `EpochOrder` | `N: Integer; ASeed, AEpoch: Int64` | a permutation of `0 .. N - 1` (`TIntArray`) | — | `N >= 0` |
| `EpochBatches` | `ASet: TTrainSet; ASeed, AEpoch: Int64; APool, ABudget: Integer` | the batches of the epoch (`TBatches`) | `ETrainData` (4.10 #8–#9) | `APool >= 1`; `ABudget >= ASet.MaxLength` |
| `SkipBatches` | `const AB: TBatches; APosition: Int64` | the batches after the first `APosition` samples | `ETrainData` (4.10 #10–#11) | `APosition` ends a batch (or is 0) |
| `BatchTensors` | `ASet: TTrainSet; const ABatch: TBatch; APad: Integer; out AIds, ATargets, AMask: TTensor` | — (the three `[B, T]` tensors in the out parameters) | `ETrainData` (4.10 #12, and #7 through `Samples`), the tch-fpc errors through `TTensor.FromArray`/`.view` | the batch non-empty, its indices in range |

Notes:

- `TTensor` is a managed record (tch-fpc): this unit never frees a
  tensor; the out tensors are owned by the caller.
- The `TStringList` objects of `TTrainSet.Create` and the parsed JSON
  documents are freed in `try … finally`; `TTrainSet`, `FSamples`,
  `TIntArray` and `TBatches` are plain dynamic arrays and need no manual
  freeing.

## 3. Data structures

- Input file: JSONL — one record per line, raw UTF-8 text. Every record
  has the string keys `id`, `dialect`, `prompt`, `answer`, `class`; other
  keys are ignored; blank lines are skipped; ids must be unique (checked
  even for records skipped by class). A `\u` escape of a character above
  U+007F is refused before the JSON parse (fpjson turns such an escape
  into `?`); T19's `qlora_dataset` writes UTF-8 directly.
- `TTrainSample` fields:

  | Field | Type | Meaning |
  |---|---|---|
  | `Id` | `string` | the record's `id` (unique in the file) |
  | `Dialect` | `string` | the record's `dialect` |
  | `Tokens` | `TTokenArray` | the rendered conversation (user `prompt`, assistant `answer`, no thinking) |
  | `LossFrom` | `Integer` | first index of `[LossFrom, LossTo)`: the trimmed answer |
  | `LossTo` | `Integer` | one past the end of the span: the answer's closing `im_end`; the newline after `im_end` is not in the loss |

- `TIntArray = array of Integer`; `TBatch = TIntArray` — indices into the
  set's samples, one batch; `TBatches = array of TBatch` — all the
  batches of an epoch, pools in order and a pool's batches in shuffled
  order.
- `TTrainSet` fields (private): `FSamples: array of TTrainSample`
  (capacity grown to `2 * N + 16`); `FSkippedClass`, `FSkippedLong:
  Integer` — the counts of the records skipped by class and by length.
- Tensor outputs of `BatchTensors` (`B` the batch size, `T` the longest
  sample of the batch):
  - `Ids` `[B, T]` int64 — the sample tokens, right-padded with `Pad`.
  - `Targets` `[B, T]` int64 — the ids shifted left by one, `Pad` at the
    end.
  - `Mask` `[B, T]` float — 1 where the target position `t + 1` is in
    `[LossFrom, LossTo)`.
- The record key `class` is used for admission only; it is not stored in
  the sample.

## 4. Processing detail

### 4.1 Types

The interface section (T15 facts, "Unit `TrainData` (interface, final)"):

```pascal
type
  ETrainData = class(Exception);

  TIntArray = array of Integer;

  TTrainSample = record
    Id: string;
    Dialect: string;
    Tokens: TTokenArray;
    LossFrom, LossTo: Integer;   // [LossFrom, LossTo) of Tokens: the answer and its im_end
  end;

  TBatch = TIntArray;            // indices into the set's samples
  TBatches = array of TBatch;

  TTrainSet = class
  private
    FSamples: array of TTrainSample;
    FSkippedClass, FSkippedLong: Integer;
    function GetSample(I: Integer): TTrainSample;
  public
    { Reads AFile; admits the records whose class is in AClasses (compared
      literally); drops the samples longer than AMaxLen tokens (counted).
      Raises ETrainData. }
    constructor Create(Tok: TQwenTokenizer; const AFile: string;
      const AClasses: array of string; AMaxLen: Integer);
    function Count: Integer;
    { The length of the longest sample (0 for none). }
    function MaxLength: Integer;
    property Samples[I: Integer]: TTrainSample read GetSample; default;
    property SkippedClass: Integer read FSkippedClass;
    property SkippedLong: Integer read FSkippedLong;
  end;

{ The tokens of the conversation user APrompt, assistant AAnswer, rendered
  without thinking; [LossFrom, LossTo) is the answer and its im_end. }
function EncodeSample(Tok: TQwenTokenizer; const APrompt, AAnswer: RawByteString;
  out LossFrom, LossTo: Integer): TTokenArray;
{ A permutation of 0 .. N - 1 from (ASeed, AEpoch) (Fisher-Yates, SplitMix64). }
function EpochOrder(N: Integer; ASeed, AEpoch: Int64): TIntArray;
{ The batches of epoch AEpoch. Raises ETrainData when APool < 1 or ABudget
  is below the longest sample. }
function EpochBatches(ASet: TTrainSet; ASeed, AEpoch: Int64; APool, ABudget: Integer): TBatches;
{ The batches after the first APosition samples of AB (APosition must end a
  batch; 0 gives AB). Raises ETrainData. }
function SkipBatches(const AB: TBatches; APosition: Int64): TBatches;
{ The samples of a batch: AIds [B, T] (int64, right-padded with APad), ATargets
  [B, T] (int64; AIds shifted left by one, APad at the end), AMask [B, T]
  (float; 1 where the target is a loss token: t + 1 in [LossFrom, LossTo)).
  T is the longest sample. Raises ETrainData for an empty batch. }
procedure BatchTensors(ASet: TTrainSet; const ABatch: TBatch; APad: Integer;
  out AIds, ATargets, AMask: TTensor);
```

Notes:

- `ETrainData` is the only exception class declared by this unit (4.10).
- `TIntArray` is this unit's own alias for a dynamic integer array
  (layout-identical to T05's `TTokenArray`).
- `TTrainSet.Samples` is the default-indexed property; `EpochBatches`
  and `BatchTensors` reach the samples through it (`ASet[I]`), so every
  index is range-checked once, in `GetSample`.
- `TBatch` and `TBatches` carry **indices**, not tokens: this is one of
  the two views T17 chooses from (external design, "Open for T17").

### 4.2 `SplitMix64`, `StateOf`, `Shuffle`

The local routines (implementation section; `SplitMix64` and `StateOf`
inside the `{$push}{$Q-}{$R-}` … `{$pop}` block):

```pascal
{$push}{$Q-}{$R-}
function SplitMix64(var S: QWord): QWord;
var
  Z: QWord;
begin
  S := S + QWord($9E3779B97F4A7C15);
  Z := S;
  Z := (Z xor (Z shr 30)) * QWord($BF58476D1CE4E5B9);
  Z := (Z xor (Z shr 27)) * QWord($94D049BB133111EB);
  Result := Z xor (Z shr 31);
end;

function StateOf(ASeed, AEpoch, APool: Int64): QWord;
begin
  Result := QWord(ASeed) + QWord(AEpoch) * QWord($D1B54A32D192ED03)
    + QWord(APool) * QWord($9E3779B97F4A7C15);
end;
{$pop}

procedure Shuffle(var A: TIntArray; AState: QWord);
var
  I, J, T: Integer;
begin
  for I := High(A) downto 1 do
  begin
    J := Integer(SplitMix64(AState) mod QWord(I + 1));
    T := A[I];
    A[I] := A[J];
    A[J] := T;
  end;
end;
```

Notes:

- `SplitMix64` is the deterministic SplitMix64 stream: advance `S` by the
  golden-ratio constant `$9E3779B97F4A7C15`, then the three standard
  avalanche steps. No libtorch random state is involved, so (seed, epoch)
  gives the same order in every process (external design, "Epoch
  order"; ITa-T15-004 compares the batches of the same epoch computed by
  two processes).
- `StateOf` folds (seed, epoch, pool) into one `QWord` with distinct
  multipliers, so every shuffle purpose has its own stream: the epoch
  permutation uses `StateOf(ASeed, AEpoch, 0)` and pool `P`'s batch
  order uses `StateOf(ASeed, AEpoch, P + 1)` (4.6, 4.7).
- `Shuffle` is the Fisher–Yates walk from `High(A)` down to 1: at each
  step a uniform `J` in `0 .. I` is drawn from the stream (modulo
  `I + 1`) and `A[I]` is swapped with `A[J]`. For an empty `A` the loop
  does not run.
- `{$Q-}{$R-}` disables range and overflow checks for `SplitMix64` and
  `StateOf` only: the `QWord` arithmetic wraps by design; the directives
  are pushed and popped around the two routines.

### 4.3 `EncodeSample`

```pascal
function EncodeSample(Tok: TQwenTokenizer; const APrompt, AAnswer: RawByteString;
  out LossFrom, LossTo: Integer): TTokenArray;
var
  M: TChatMessages;
  O: TChatOptions;
begin
  M := nil;
  SetLength(M, 2);
  M[0] := ChatMsg(crUser, APrompt);
  M[1] := ChatMsg(crAssistant, AAnswer);
  O := DefaultChatOptions;
  O.EnableThinking := False;
  Result := EncodeTraining(Tok, M, O, LossFrom, LossTo);
end;
```

Notes:

- The conversation has exactly two messages — user `APrompt`, assistant
  `AAnswer`; there is no system message (external design, "Rendering
  and loss span", DESIGN 8.14 "Thinking format").
- `O.EnableThinking := False`: `EncodeTraining` renders the empty think
  block before the answer (DESIGN 8.14).
- `LossFrom`/`LossTo` come out of `EncodeTraining` (T07): the span
  `[LossFrom, LossTo)` is the trimmed answer and its closing `im_end`;
  the newline after `im_end` is not in the loss.
- There is no token cache (external design, "No token cache"): on the
  rehearsal tokenizer a 2000-token sample tokenises in about 4 ms, so
  the 4000 samples of data set v1 take seconds at every start and
  `TTrainSet.Create` tokenises every admitted record (4.5).

### 4.4 `HasWideEscape`, `GetStr`

```pascal
{ A \u escape of a character above #127 anywhere in the line. }
function HasWideEscape(const L: RawByteString): Boolean;
var
  I, V: Integer;
begin
  Result := False;
  I := 1;
  while I < Length(L) do
  begin
    if L[I] = '\' then
    begin
      if (L[I + 1] = 'u') and (I + 5 <= Length(L))
        and TryStrToInt('$' + Copy(L, I + 2, 4), V) and (V > 127) then
        Exit(True);
      Inc(I, 2);
    end
    else
      Inc(I);
  end;
end;

function GetStr(O: TJSONObject; const AKey: string; ALine: Integer): RawByteString;
var
  D: TJSONData;
begin
  D := O.Find(AKey);
  if (D = nil) or (D.JSONType <> jtString) then
    raise ETrainData.CreateFmt('line %d: missing or bad %s', [ALine, AKey]);
  Result := D.AsString;
end;
```

Notes on `HasWideEscape`:

- Scans the line byte by byte. A backslash whose next byte is `u`, with
  the four hex digits `L[I + 2 .. I + 5]` present and parsing to a value
  `V > 127`, is a `\u` escape of a character above U+007F and returns
  `True` — the case fpjson would turn into `?` (external design, "Record
  format").
- A backslash always skips two bytes: an escaped backslash (`\\`) is
  text, so `\` + `u00e9` with the first backslash escaped does not
  trigger; `\u0041` parses to `65 <= 127` and does not trigger either
  (only `V > 127` is refused).
- The check runs **before** the JSON parse, so the refusal does not
  depend on what the parser would do with the escape.

Notes on `GetStr`:

- `O.Find(AKey)` must return a JSON string; a missing key or a value of
  another JSON type (number, `null`, array, …) raises `line %d: missing
  or bad <key>` with the 1-based line number `ALine` (4.10 #5).

### 4.5 `TTrainSet`

#### `Create`

```pascal
constructor TTrainSet.Create(Tok: TQwenTokenizer; const AFile: string;
  const AClasses: array of string; AMaxLen: Integer);
var
  L, Ids: TStringList;
  I, N, J: Integer;
  D: TJSONData;
  O: TJSONObject;
  S: TTrainSample;
  Cls, Prompt, Answer: RawByteString;
  Admit: Boolean;
begin
  inherited Create;
  if AMaxLen < 1 then
    raise ETrainData.CreateFmt('max length must be positive, got %d', [AMaxLen]);
  if not FileExists(AFile) then
    raise ETrainData.Create('no data file ' + AFile);
  L := TStringList.Create;
  Ids := TStringList.Create;
  try
    Ids.Sorted := True;
    L.LoadFromFile(AFile);
    N := 0;
    for I := 0 to L.Count - 1 do
    begin
      if Trim(L[I]) = '' then
        Continue;
      if HasWideEscape(L[I]) then
        raise ETrainData.CreateFmt('line %d: a \u escape of a non-ASCII character; write UTF-8', [I + 1]);
      try
        D := GetJSON(L[I], False);
      except
        on E: EJSONParser do
          raise ETrainData.CreateFmt('line %d: not a JSON object', [I + 1]);
      end;
      try
        if not (D is TJSONObject) then
          raise ETrainData.CreateFmt('line %d: not a JSON object', [I + 1]);
        O := TJSONObject(D);
        S.Id := GetStr(O, 'id', I + 1);
        S.Dialect := GetStr(O, 'dialect', I + 1);
        Prompt := GetStr(O, 'prompt', I + 1);
        Answer := GetStr(O, 'answer', I + 1);
        Cls := GetStr(O, 'class', I + 1);
        if Ids.IndexOf(S.Id) >= 0 then
          raise ETrainData.CreateFmt('line %d: duplicate id %s', [I + 1, S.Id]);
        Ids.Add(S.Id);
        Admit := False;
        for J := 0 to High(AClasses) do
          if AClasses[J] = Cls then
            Admit := True;
        if not Admit then
        begin
          Inc(FSkippedClass);
          Continue;
        end;
        S.Tokens := EncodeSample(Tok, Prompt, Answer, S.LossFrom, S.LossTo);
      finally
        D.Free;
      end;
      if Length(S.Tokens) > AMaxLen then
      begin
        Inc(FSkippedLong);
        Continue;
      end;
      if N = Length(FSamples) then
        SetLength(FSamples, 2 * N + 16);
      FSamples[N] := S;
      Inc(N);
    end;
    SetLength(FSamples, N);
  finally
    Ids.Free;
    L.Free;
  end;
end;
```

Notes:

- The checks run in this order (4.10 #1–#6): `AMaxLen` positive first;
  the file exists; then, per non-blank line (1-based line numbers), the
  `\u` escape refusal, the parse, the object check, the five keys in the
  order `id`, `dialect`, `prompt`, `answer`, `class`, and the duplicate
  id.
- Per-line order: skip blank lines; refuse the wide escape; parse (`EJSONParser`
  is converted to `line %d: not a JSON object`; any other exception of
  `GetJSON` passes through unchanged); require a JSON object; read the
  five string keys; refuse a duplicate id — **before** the class check,
  so a record skipped by class still counts toward the uniqueness; admit
  by class (each `AClasses[J]` compared literally with `Cls`; an empty
  list admits nothing); only admitted records are tokenised; drop the
  samples **longer than** `AMaxLen` — a sample of exactly `AMaxLen`
  tokens is admitted (the condition is `> AMaxLen`); store the sample.
- `FSkippedClass` and `FSkippedLong` count the refusals; both are read
  through the properties (4.10 #6 covers the duplicate id, not the
  skips).
- `Ids` is a sorted `TStringList`; `IndexOf` + `Add` are the uniqueness
  check and registry.
- The `FSamples` capacity doubles plus 16 when full
  (`SetLength(FSamples, 2 * N + 16)`); the array is cut to `N` lines at
  the end.
- `GetJSON(L[I], False)` keeps the UTF-8 bytes of the line (4.4).
- The two `TStringList`s and every parsed JSON document are freed in
  `finally`; the sample is stored only when it was admitted by both the
  class and the length.

#### `Count`, `MaxLength`, `GetSample`

```pascal
function TTrainSet.Count: Integer;
begin
  Result := Length(FSamples);
end;

function TTrainSet.MaxLength: Integer;
var
  I: Integer;
begin
  Result := 0;
  for I := 0 to High(FSamples) do
    Result := Max(Result, Length(FSamples[I].Tokens));
end;

function TTrainSet.GetSample(I: Integer): TTrainSample;
begin
  if (I < 0) or (I >= Length(FSamples)) then
    raise ETrainData.CreateFmt('sample %d outside 0..%d', [I, High(FSamples)]);
  Result := FSamples[I];
end;
```

Notes:

- `Count` is the number of admitted samples; `MaxLength` is 0 for an
  empty set (the loop is empty when `High(FSamples) = -1`).
- `GetSample` is the single accessor: `Samples[I]`, and every
  `ASet[...]` of `EpochBatches` and `BatchTensors`, goes through it, so
  the range check `sample %d outside 0..%d` (with the upper bound
  `Count - 1 = High(FSamples)`) is raised in exactly one place (4.10
  #7).

### 4.6 `EpochOrder`

```pascal
function EpochOrder(N: Integer; ASeed, AEpoch: Int64): TIntArray;
var
  I: Integer;
begin
  Result := nil;
  SetLength(Result, N);
  for I := 0 to N - 1 do
    Result[I] := I;
  Shuffle(Result, StateOf(ASeed, AEpoch, 0));
end;
```

Notes:

- The identity permutation is shuffled with the stream
  `StateOf(ASeed, AEpoch, 0)` (pool argument 0; the pool streams of
  4.7 start at `P + 1`, so they never collide with this one).
- Same `(N, seed, epoch)` gives the same order in every process — no
  libtorch random state (external design, "Epoch order").
- For `N = 0` the array is empty and the shuffle is a no-op.

### 4.7 `EpochBatches`

```pascal
function EpochBatches(ASet: TTrainSet; ASeed, AEpoch: Int64; APool, ABudget: Integer): TBatches;
var
  Order, Pool, Len, Perm: TIntArray;
  Cur: TBatch;
  PoolB: TBatches;
  P, I, J, K, NB, Lo, Hi, T: Integer;
begin
  if APool < 1 then
    raise ETrainData.CreateFmt('pool must be positive, got %d', [APool]);
  if ABudget < ASet.MaxLength then
    raise ETrainData.CreateFmt('budget %d is below the longest sample, %d', [ABudget, ASet.MaxLength]);
  Result := nil;
  NB := 0;
  Order := EpochOrder(ASet.Count, ASeed, AEpoch);
  P := 0;
  while P * APool < Length(Order) do
  begin
    Lo := P * APool;
    Hi := Min(Lo + APool, Length(Order)) - 1;
    { the pool sorted by length, ties in the epoch's order (insertion sort) }
    Pool := Copy(Order, Lo, Hi - Lo + 1);
    Len := nil;
    SetLength(Len, Length(Pool));
    for I := 0 to High(Pool) do
      Len[I] := Length(ASet[Pool[I]].Tokens);
    for I := 1 to High(Pool) do
    begin
      J := I;
      while (J > 0) and (Len[J - 1] > Len[J]) do
      begin
        T := Len[J]; Len[J] := Len[J - 1]; Len[J - 1] := T;
        T := Pool[J]; Pool[J] := Pool[J - 1]; Pool[J - 1] := T;
        Dec(J);
      end;
    end;
    { greedy: ascending lengths, so the newest sample is the longest }
    PoolB := nil;
    Cur := nil;
    for I := 0 to High(Pool) do
    begin
      if (Length(Cur) + 1) * Len[I] > ABudget then
      begin
        SetLength(PoolB, Length(PoolB) + 1);
        PoolB[High(PoolB)] := Cur;
        Cur := nil;
      end;
      SetLength(Cur, Length(Cur) + 1);
      Cur[High(Cur)] := Pool[I];
    end;
    SetLength(PoolB, Length(PoolB) + 1);
    PoolB[High(PoolB)] := Cur;
    { the pool's batches in a shuffled order }
    Perm := nil;
    SetLength(Perm, Length(PoolB));
    for K := 0 to High(Perm) do
      Perm[K] := K;
    Shuffle(Perm, StateOf(ASeed, AEpoch, P + 1));
    SetLength(Result, NB + Length(PoolB));
    for K := 0 to High(Perm) do
      Result[NB + K] := PoolB[Perm[K]];
    Inc(NB, Length(PoolB));
    Inc(P);
  end;
end;
```

Notes:

- The checks come first, in this order (4.10 #8–#9): `APool < 1`, then
  `ABudget < ASet.MaxLength`.
- The order of `EpochOrder(ASet.Count, ASeed, AEpoch)` is cut into pools
  of `APool` samples; the last pool is shorter when the count is not a
  multiple of `APool`.
- Each pool is sorted by sample length with an insertion sort carrying
  `Len` and `Pool` together; the shift condition is the strict
  `Len[J - 1] > Len[J]`, so the sort is **stable**: equal lengths keep
  the epoch's order.
- The greedy cut walks the pool in ascending lengths, so the last
  (newest) sample of the current batch is the longest one: a new batch
  starts when `(Length(Cur) + 1) * Len[I] > ABudget`. A batch whose
  count × longest is exactly `ABudget` is admitted; every admitted batch
  fits the budget and covers a prefix of the pool; the last partial
  batch is pushed after the loop.
- The pool's batches are reordered by
  `Shuffle(Perm, StateOf(ASeed, AEpoch, P + 1))` — one stream per pool
  (4.2) — and appended to `Result` in pool order; `NB` tracks the
  running total.
- The result depends only on the set, the seed, the epoch, the pool and
  the budget: a second call is identical, a different epoch or seed
  differs (external design, "Batches (pools and budget)").
- An empty set yields no batches (`Result` stays `nil`).

### 4.8 `SkipBatches`

```pascal
function SkipBatches(const AB: TBatches; APosition: Int64): TBatches;
var
  I: Integer;
  Done: Int64;
begin
  if APosition < 0 then
    raise ETrainData.CreateFmt('position %d is negative', [APosition]);
  I := 0;
  Done := 0;
  while (Done < APosition) and (I < Length(AB)) do
  begin
    Inc(Done, Length(AB[I]));
    Inc(I);
  end;
  if Done <> APosition then
    raise ETrainData.CreateFmt('position %d is not at the end of a batch', [APosition]);
  Result := Copy(AB, I, Length(AB) - I);
end;
```

Notes:

- `APosition` counts the samples of the epoch's order already consumed —
  T16's checkpoint `Position`; only values that end a batch are valid
  (0 ends before the first batch and returns `AB` unchanged, since the
  loop does not run and `Copy(AB, 0, Length(AB))` is a copy).
- The walk stops when `Done >= APosition` or the batches are exhausted;
  `Done <> APosition` then rejects both a position inside a batch and a
  position beyond the last batch (4.10 #10–#11).
- Because every batch is a function of (seed, epoch) (4.7), skipping the
  consumed batches is an exact resumption of the run (external design,
  "Batches (pools and budget)"; system flow, the T17 caller).

### 4.9 `BatchTensors`

```pascal
procedure BatchTensors(ASet: TTrainSet; const ABatch: TBatch; APad: Integer;
  out AIds, ATargets, AMask: TTensor);
var
  B, T, I, J, N: Integer;
  Ids, Tg: array of Int64;
  M: array of Double;
  S: TTrainSample;
begin
  B := Length(ABatch);
  if B = 0 then
    raise ETrainData.Create('empty batch');
  T := 0;
  for I := 0 to B - 1 do
    T := Max(T, Length(ASet[ABatch[I]].Tokens));
  Ids := nil;
  Tg := nil;
  M := nil;
  SetLength(Ids, B * T);
  SetLength(Tg, B * T);
  SetLength(M, B * T);
  for I := 0 to B - 1 do
  begin
    S := ASet[ABatch[I]];
    N := Length(S.Tokens);
    for J := 0 to T - 1 do
    begin
      if J < N then
        Ids[I * T + J] := S.Tokens[J]
      else
        Ids[I * T + J] := APad;
      if J + 1 < N then
        Tg[I * T + J] := S.Tokens[J + 1]
      else
        Tg[I * T + J] := APad;
      if (J + 1 >= S.LossFrom) and (J + 1 < S.LossTo) then
        M[I * T + J] := 1
      else
        M[I * T + J] := 0;
    end;
  end;
  AIds := TTensor.FromArray(Ids).view([B, T]);
  ATargets := TTensor.FromArray(Tg).view([B, T]);
  AMask := TTensor.FromArray(M).view([B, T]);
end;
```

Notes:

- An empty batch refuses first (`empty batch`, 4.10 #12); the length of
  `T` is computed through `ASet[ABatch[I]]`, so an out-of-range index
  raises the `GetSample` message (4.10 #7) before anything is built.
- `T` is the longest sample of the batch; the three flat arrays are all
  of size `B * T`, filled row `I`, column `J` (flat index `I * T + J`):
  - `Ids`: `S.Tokens[J]` while `J < N` (= the sample's length), else
    `APad` — right padding.
  - `Targets`: `S.Tokens[J + 1]` while `J + 1 < N`, else `APad` — the ids
    shifted left by one; the last column of every row (and every pad
    position) is `APad`.
  - `Mask`: 1 exactly where the **target** index `J + 1` lies in
    `[S.LossFrom, S.LossTo)` — the trimmed answer and the closing
    `im_end`; the newline after `im_end` and every prompt token are 0.
  - A mask entry of a pad position is 0 because `J + 1 >= N` and
    `LossTo <= N`.
- The sources are `array of Int64` (→ int64 tensors) and `array of
  Double` (→ float tensor); `TTensor.FromArray(...).view([B, T])`
  builds each out parameter (tch-fpc: the tensors are managed records;
  this unit does not free them — section 2, notes).
- The loss of DESIGN 8.14 — the sum of the masked cross-entropies
  divided by the mask count — is computed by T17 from `Ids`, `Targets`
  and `Mask`; this unit only renders the tensors (external design,
  "Tensors").

### 4.10 Checks and messages (in order)

All raised as `ETrainData` (unit `TrainData`) unless the row says
otherwise, in this order (exact messages from the T15 facts):

| # | Where | Condition (checked in this order) | Message |
|---|---|---|---|
| 1 | `TTrainSet.Create` | `AMaxLen < 1` | `max length must be positive, got %d` (the `AMaxLen`) |
| 2 | `TTrainSet.Create` | the file does not exist | `no data file <file>` (the file name as given) |
| 3 | `TTrainSet.Create`, per non-blank line (1-based line numbers of the file) | a `\u` escape of a character above U+007F anywhere in the line | `line %d: a \u escape of a non-ASCII character; write UTF-8` (the backslash-u written literally in the message) |
| 4 | `TTrainSet.Create` | the line does not parse (`EJSONParser`) or is not a JSON object | `line %d: not a JSON object` |
| 5 | `TTrainSet.Create`, for the keys `id`, `dialect`, `prompt`, `answer`, `class` in this order | the key is missing or not a string (a number, `null`, an array, …) | `line %d: missing or bad <key>` |
| 6 | `TTrainSet.Create` | the id already occurred on an earlier line (checked before the class, so skipped records count too) | `line %d: duplicate id <id>` |
| 7 | `Samples[I]` (and `BatchTensors` through it) | `(I < 0) or (I >= Count)` | `sample %d outside 0..%d` (the `I` and `Count - 1`) |
| 8 | `EpochBatches` | `APool < 1` | `pool must be positive, got %d` (the `APool`) |
| 9 | `EpochBatches` | `ABudget < ASet.MaxLength` | `budget %d is below the longest sample, %d` (the `ABudget` and the `MaxLength`) |
| 10 | `SkipBatches` | `APosition < 0` | `position %d is negative` |
| 11 | `SkipBatches` | the running sum of batch sizes does not end exactly at `APosition` (also beyond the last batch) | `position %d is not at the end of a batch` |
| 12 | `BatchTensors` | the batch is empty (length 0) | `empty batch` |
| — | from QwenChat (passed on) | none for these records — every record has a user and an assistant message | — |

Notes:

- A UT error line catches the named exception class and compares the
  whole message; the line fails when no exception comes (T15 facts,
  "Unit tests").
- The exceptions that are not `ETrainData` pass through unchanged: the
  other `GetJSON` errors (only `EJSONParser` is converted, 4.5), the
  `QwenChat` errors of `EncodeTraining` (none for these records), and
  the tch-fpc errors of `TTensor.FromArray`/`.view` (4.9).

### 4.11 Business rules of the external design and where they are handled

| Business rule (`T15-external.md`) | Handled by |
|---|---|
| JSONL, one record per line; the string keys `id`, `dialect`, `prompt`, `answer`, `class`; other keys ignored; blank lines skipped; ids unique | `TTrainSet.Create` (4.5), `GetStr` (4.4), 4.10 #4–#6 |
| raw UTF-8; a `\u` escape of a character above U+007F refused (fpjson would turn it into `?`) | `HasWideEscape` (4.4), `Create` (4.5), 4.10 #3 |
| admission: the `class` is in the admitted list (compared literally; default `compiled-run`, `fpc-approx`); samples longer than the maximum dropped; the skips counted separately | `Create` (4.5: `FSkippedClass`/`FSkippedLong`, literal compare, `> AMaxLen`), 4.10 #6 |
| rendering without thinking: user = prompt, assistant = answer, no system message, empty think block before the answer; loss span = trimmed answer + its closing `im_end`; the newline after `im_end` not in the loss | `EncodeSample` (4.3) through `EncodeTraining` (T07); the span stored per sample (4.1) |
| no token cache (a 2000-token sample tokenises in 4 ms; the 4000 samples of data set v1 in seconds) | no cache anywhere: every `Create` tokenises its admitted records (4.5); the samples stay in `FSamples` (4.1) |
| epoch order: Fisher–Yates with SplitMix64 from (seed, epoch); no libtorch random state; the same in every process | `SplitMix64`/`StateOf`/`Shuffle` (4.2), `EpochOrder` (4.6) |
| batches: pools of `Pool`; each pool sorted by length (ties keep the epoch's order); greedy cut with count × longest ≤ `Budget`; the pool's batches shuffled (SplitMix64 from seed, epoch, pool); every batch a function of (seed, epoch) | `EpochBatches` (4.7) |
| resumption: skip the batches of the already-consumed samples (T16's `Position`) | `SkipBatches` (4.8), 4.10 #10–#11 |
| tensors: `Ids` [B, T] int64 right-padded with `Pad`; `Targets` [B, T] int64, ids shifted left by one, `Pad` at the end; `Mask` [B, T] float, 1 where the target `t + 1` is in `[LossFrom, LossTo)`; T the longest sample | `BatchTensors` (4.9); the `Pad` token id is supplied by the caller |
| the loss of DESIGN 8.14 (sum of the masked cross-entropies / mask count) | `BatchTensors` delivers `Ids`/`Targets`/`Mask`; the division by the mask count is in T17 (system flow, the T17 caller) |
| open for T17: the batch both as sample indices and as padded tensors | `TBatch`/`TBatches` (4.1) plus `BatchTensors` (4.9) |
| the error list (external design, "Errors") | 4.10 #1–#12 |

## 5. Change history

| Date | Change | Reason |
|---|---|---|
| T15a (initial) | First edition, from the external design, the system flow and the T15 facts. | new unit |
