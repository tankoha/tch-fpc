# T15 facts (checked by the harness owner)

T15 is the unit `TrainData` (DESIGN 8.14 "Batches", "Loss mask",
"Thinking format", 8.12, 8.16): the data set file becomes training
samples (tokens and the loss span), an epoch becomes batches within a
position budget, and a batch becomes tensors (ids, targets, mask).
Pascal plus `Tch` for the tensors; no model.

DESIGN 8.18's acceptance for T15: "masks cover exactly the answer;
buckets respect the position budget". Decisions taken by the harness
owner for this task (DESIGN 8.12–8.16 leave them open):
- **Records**: JSONL, one record per line with the string keys `id`,
  `dialect`, `prompt`, `answer` and `class` (the verification class of
  8.16; T19's `qlora_dataset` writes it into every record). Other keys
  are ignored; blank lines are skipped; ids must be unique. Text is raw
  UTF-8: fpjson (read with `GetJSON(Line, False)`, which keeps UTF-8
  bytes) turns a `\u` escape of a non-ASCII character into `?`
  (measured), so such an escape is refused before parsing; T19 writes
  UTF-8 directly.
- **Admission**: a record is admitted when its class is in the list given
  (the training configuration's admitted classes, default `compiled-run`,
  `fpc-approx`; 8.16); a sample longer than the maximum length (the
  budget, 2048) is dropped. Both are counted.
- **Rendering**: user = prompt, assistant = answer, rendered without
  thinking (`QwenChat.EncodeTraining` with `EnableThinking` False: no
  system message, the empty think block before the answer; 8.14
  "Thinking format"). The loss covers the trimmed answer and its closing
  `im_end`, `[LossFrom, LossTo)` of T07; the newline after `im_end` is not
  in the loss.
- **No token cache.** DESIGN 8.12 planned one; measured on the rehearsal
  tokenizer, `EncodeTraining` of a 2000-token sample takes 4 ms, so the
  4000 samples of data set v1 tokenise in seconds at every start.
- **Epoch order**: a permutation from (seed, epoch) by Fisher–Yates with
  SplitMix64 (no libtorch random state), so it is the same in every
  process.
- **Batches**: the order is cut into pools of `Pool` samples; each pool is
  sorted by length (ties keep the epoch's order) and cut greedily into
  batches with count × the longest ≤ `Budget`; the batches of a pool are
  shuffled (SplitMix64 from seed, epoch, pool). Every batch is a function
  of (seed, epoch), so a run resumes by skipping the batches of the
  samples already consumed (`SkipBatches`, T16's `Position`).
- **Tensors**: `Ids` [B, T] int64 right-padded with `Pad`; `Targets`
  [B, T] int64, `Ids` shifted left by one (`Pad` at the end); `Mask`
  [B, T] float, 1 where the target `t + 1` is in `[LossFrom, LossTo)`. T
  is the longest sample of the batch. The loss of 8.14 is the sum of the
  masked cross-entropies divided by the mask count.
- **Open for T17** (not T15's): `QwenModel` (T11) runs one sequence
  `[T, H]`; a batch of B samples needs either B forwards (each paying the
  dequantisation) or a batched model. T15 delivers both views: the batch
  as sample indices and as padded tensors.

Measured by the harness owner with a prototype written exactly as below:
- UT: every row of the tables below passes; the unit then has 100 % line
  coverage; mutants are caught (the mask one position early: 108, 109; a batch exactly at the budget refused: 103; pools not sorted: 103, 104; the batches of a pool not shuffled: 103; `SkipBatches` accepting a position inside a batch: 107; `HasWideEscape` not skipping an escaped backslash: 011; every class admitted: 004–006, 013, 014, 108, 110; a sample exactly at the maximum dropped: 005; duplicate ids not refused: 010).
- ITa on the rehearsal model's tokenizer: 0, 0, 0, 0, 0; 37 samples admitted (3 skipped by class, 2 too long), 33 batches over three epochs, 3.4 samples per batch, 96 % of the padded positions are tokens (cases below);
  with the mask one position early 005 is NG (37 rows), with every class admitted 002, with unsorted pools 003 (16 batches over the budget); with the skeleton every case is NG.

These facts are final; do not investigate further or write probes.

## Unit `TrainData` (interface, final; in the skeleton)

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

## Implementation of `TrainData` (write it exactly so)

The routines in the order of the unit. The local routines (`SplitMix64`,
`StateOf` inside `{$push}{$Q-}{$R-}` ... `{$pop}`, `Shuffle`,
`HasWideEscape`, `GetStr`) are not in the interface; write them in the
implementation section where shown.

```pascal
implementation

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

{ TTrainSet }

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

## Checks and messages (in this order)

`ETrainData` (unit `TrainData`):
- `TTrainSet.Create`: `CreateFmt('max length must be positive, got %d',
  [AMaxLen])`; `no data file <file>`; then per non-blank line (1-based
  line numbers of the file): `CreateFmt('line %d: a \u escape of a
  non-ASCII character; write UTF-8', ...)` (the backslash-u written
  literally in the message); `line %d: not a JSON object` (a parse error
  or not an object); `line %d: missing or bad <key>` for `id`, `dialect`,
  `prompt`, `answer`, `class` in this order (missing or not a string);
  `line %d: duplicate id <id>` (checked before the class, so skipped
  records count too).
- `Samples[I]` (and `BatchTensors` through it): `CreateFmt('sample %d
  outside 0..%d', [I, Count - 1])`.
- `EpochBatches`: `CreateFmt('pool must be positive, got %d', [APool])`;
  `CreateFmt('budget %d is below the longest sample, %d', [ABudget,
  MaxLength])`.
- `SkipBatches`: `CreateFmt('position %d is negative', ...)`;
  `CreateFmt('position %d is not at the end of a batch', ...)` (also
  beyond the last batch).
- `BatchTensors`: `empty batch`.
- From QwenChat (passed on): none for these records (every record has a
  user and an assistant message).

## UT fixture and helpers (harness owner's; nobody changes them)

`tests/ut/TrainData_test.pas` (skeleton: helpers, runner, no test
methods). The tokenizer is T05's tiny vocabulary
(`TokenizerFixture.NewTinyTokenizer`): bytes are tokens except the merges
of T05 and the specials; `im_start`, `im_end` and `think` are one token
each, `</think>` is 8 byte tokens. With prompt P and answer A free of
merges, a sample has 34 + |P| + |trimmed A| tokens, LossFrom = 32 + |P|,
LossTo = LossFrom + |trimmed A| + 1.
- Constants `TagS`, `TagE`, `ThinkO`, `ThinkC` (the tags, built from
  pieces) and `BS` (a backslash; JSON escapes are built from it, never
  written literally); `FixLines`: the four records `a` (objfpc, `Q`/`A`,
  compiled-run), `b` (delphi, `xyz`/`  12345  `, fpc-approx, an extra key),
  `c` (oxygene, static-only), `d` (objfpc, prompt 日本 and answer é in
  UTF-8, compiled-run).
- `Tok` (created once); `WriteLines(Name, Lines)` (a file under the test's
  temporary directory, each line ended by #10; returns its path);
  `FixSet(MaxLen)` (FixLines with the classes compiled-run and
  fpc-approx); `LenSet(Lens)` (sample I has exactly Lens[I] tokens: id
  `sI`, prompt `P`, answer of Lens[I] − 35 `z`); `BatchText(B)` (indices
  by spaces, batches by `|`); `IntsText(A)`; `Arr(T)`.
- `CheckLoad(Self, Lines, MaxLen, Message)` (creating a set of Lines with
  the class compiled-run raises ETrainData with Message); `Span(S, From,
  To)` (the decoded tokens of sample S in [From, To)); `CheckBatches(Self,
  Set, Batches, Budget)` (every batch non-empty and within the budget,
  every sample exactly once).
- Each test frees what it creates in `try ... finally`.

## Unit tests (rows; every row has one method, every method belongs to a row)

Error rows catch the named exception class and compare the whole message
(`AssertEquals`); a row fails if no exception comes. "exact" means equal
strings or equal numbers.

`TrainData`, stage 1 (task T15b1): `EncodeSample`, `TTrainSet`; 001–014.

| Row | Input | Expected |
|---|---|---|
| 001 | `EncodeSample(Tok, 'Q', 'A', From, To)` | 36 tokens; From 33, To 35; the decoded text `TagS + 'user'#10'Q' + TagE + #10 + TagS + 'assistant'#10 + ThinkO + #10#10 + ThinkC + #10#10'A' + TagE + #10`; tokens [From, To) decode to `'A' + TagE` |
| 002 | `EncodeSample(Tok, 'xyz', '  12345  ', ..)` | 42 tokens; From 35, To 41; [From, To) decodes to `'12345' + TagE` (the answer trimmed) |
| 003 | `EncodeSample(Tok, 日本, é, ..)` (UTF-8) | 42 tokens; From 38, To 41; [From, To) decodes to `é + TagE` |
| 004 | `FixSet(100)` | Count 3; SkippedClass 1; SkippedLong 0; ids `a b d`; dialects `objfpc delphi objfpc`; lengths `36 42 42`; LossFrom/LossTo `33 35 35 41 38 41`; MaxLength 42 |
| 005 | `FixSet(41)`; then `FixSet(42)` | Count 1 (`a`), SkippedLong 2, SkippedClass 1; at 42: Count 3 |
| 006 | lines `''`, FixLines[0], `'   '`, FixLines[1], `''` with both classes; then FixLines with the classes `[]` | Count 2, ids `a b`; with no class: Count 0, SkippedClass 4, MaxLength 0 |
| 007 | CheckLoad of [FixLines[0], a record without `answer`]; of a record without `id`; of one with `"class":1` | ETrainData `line 2: missing or bad answer`; `line 1: missing or bad id`; `line 1: missing or bad class` |
| 008 | `"id":7`; `"dialect":null`; `"prompt":["p"]` (with `"class":"x"`) | ETrainData `line 1: missing or bad id`; `line 1: missing or bad dialect`; `line 1: missing or bad prompt` |
| 009 | `{"id":` ; [`''`, `[1, 2]`] | ETrainData `line 1: not a JSON object`; `line 2: not a JSON object` |
| 010 | [FixLines[0], FixLines[2], FixLines[0]] | ETrainData `line 3: duplicate id a` |
| 011 | answer `BS + 'u00e9'`; then answer `BS + 'u0041' + BS + BS + 'u00e9' + BS + 'n'` | ETrainData `line 1: a \u escape of a non-ASCII character; write UTF-8` (the backslash as `BS`); the second loads: Count 1, the loss span decodes to `'A' + BS + 'u00e9' + TagE` (the escaped backslash is text; the trailing newline trimmed) |
| 012 | a missing file `<tmp>missing.jsonl`; MaxLen 0 | ETrainData `no data file <tmp>missing.jsonl`; `max length must be positive, got 0` |
| 013 | `FixSet(100)`: `S[3]`; `S[-1]` | ETrainData `sample 3 outside 0..2`; `sample -1 outside 0..2` |
| 014 | `LenSet([36, 40, 77])`; `FixSet(100)` | every loss span decodes to the `z`s (Length − 35 of them) + TagE; lengths `36 40 77`; FixSet: spans `'A'`, `'12345'`, `é`, each + TagE, and the tokens after the span decode to `#10` |

`TrainData`, stage 2 (task T15b2): `EpochOrder`, `EpochBatches`,
`SkipBatches`, `BatchTensors`; 101–110.

| Row | Input | Expected |
|---|---|---|
| 101 | `EpochOrder(10, 1, 0)`; `(10, 1, 1)`; `(10, 2, 0)` twice; `(0, 1, 0)`; `(1, 1, 0)` | `4 2 8 1 9 3 0 6 7 5`; `8 0 4 2 7 6 1 3 9 5`; seed 2 equal twice and differs from seed 1; ''; `0` |
| 102 | `EpochOrder(1000, -5, 3)` | 1000 entries, each of 0..999 once; differs from `EpochOrder(1000, 0, 0)` |
| 103 | `LenSet([40, 50, 60, 70, 80, 90, 100, 110])`: `EpochOrder(8, 3, 0)`; `EpochBatches(S, 3, 0, 4, 200)`; `EpochBatches(S, 3, 0, 8, 110)` | order `7 0 1 4 2 6 3 5`; batches `7\|4\|0 1\|2 3\|5 6` (`CheckBatches` at 200); at 110: 7 batches (40 and 50 together), `CheckBatches` at 110 |
| 104 | `LenSet` of 200 samples, length 36 + (37 I mod 265): `EpochBatches(S, 11, 2, 16, 512)` | `CheckBatches` at 512; every batch from one pool (positions I div 16 of `EpochOrder(200, 11, 2)`), pools in order; equal on a second call; differs for epoch 3; pool 1000, budget 2048: `CheckBatches` |
| 105 | `LenSet([40, 110])`: pool 0; budget 109 | ETrainData `pool must be positive, got 0`; `budget 109 is below the longest sample, 110` |
| 106 | an empty file: `EpochBatches(S, 1, 0, 4, 200)` | no batches |
| 107 | batches `[4 1] [0] [2 3 5]`: `SkipBatches` at 0, 3, 6, 1, 7, −1 | `4 1\|0\|2 3 5`; `2 3 5`; none; ETrainData `position 1 is not at the end of a batch`; `position 7 is not at the end of a batch`; `position -1 is negative` |
| 108 | `FixSet(100)`: `BatchTensors(S, [0, 2], 7, ..)` | shape `2 42`; kinds int64, int64, float; row 0: the 36 tokens then 7 × 6; row 1: the 42 tokens; targets row 0: tokens 1..35, then 7 (from column 35); row 1's last target 7; mask ones at flat indices `32 33 79 80 81` |
| 109 | `LenSet([36, 40, 50, 77, 120, 121])`: every batch of `EpochBatches(S, 9, 0, 6, 250)` through `BatchTensors(.., 0, ..)` | in every row the targets under the mask decode to the row's `z`s + TagE |
| 110 | `FixSet(100)`: `BatchTensors(S, nil, ..)`; `BatchTensors(S, [0, 3], ..)` | ETrainData `empty batch`; `sample 3 outside 0..2` |

Build and run (the GPUs hidden as in the gate):
`fpc -Mobjfpc -Sh -vew -l- -Fu{proj}/src -Fu{proj}/../../src -Fu{proj}/tests/ut -FU{proj}/work/bin -Fl{proj}/../../build/libtch-rocm -k-rpath={proj}/../../build/libtch-rocm {proj}/tests/ut/TrainData_test.pas -o{proj}/work/bin/TrainData_test`
then `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 {proj}/work/bin/TrainData_test --all --format=plain`.
Warnings from other units (`GgufQuant`, `GgufQuantT`, `QwenTokenizer`,
`generics.dictionaries.inc`) and libtorch's GPU warnings on stderr are
expected. Until a stage is written, the test program without test
methods reports `No valid tests found`; that is expected.

## ITa (prepared by the harness owner)

`QLORA_MODEL=<rehearsal GGUF> {proj}/tests/ita/T15-run.sh` builds
`tests/ita/T15/data_check.pas` against `src/` and runs it from a fresh
directory outside the project with the rehearsal model's tokenizer on
`tests/ita/T15/records.jsonl` (committed; made by
`tests/ita/T15/make-records.bb` from pieces of this sample's sources: 42
records, prompts in English and Japanese, three of class static-only, two
longer than 2048 tokens): once `check`, twice `dump` in two processes.
Classes compiled-run and fpc-approx, maximum length 2048, seed 15, pool
64, budget 2048, epochs 0–2. Writes `{proj}/work/ita/T15/summary.tsv`
(case, OK/NG, value, limit, note; then the exit codes and the count of
stderr lines other than the GPU warnings). OK when value ≤ limit. Under a
minute.

| Case | Compared | Limit |
|---|---|---|
| ITa-T15-001 | every sample: its tokens against `Encode` of the text rendered by the driver from the template's pieces; the decoded parts before, inside and after `[LossFrom, LossTo)` against the prompt part, the trimmed answer + im_end, one newline (count of mismatches) | 0 |
| ITa-T15-002 | admitted, skipped by class and too long against the counts from the file (count of mismatches) | 0 |
| ITa-T15-003 | epochs 0–2: batches empty or over the budget, samples not seen exactly once per epoch (count) | 0 |
| ITa-T15-004 | the batches of epochs 0–2 written by two processes (0 when equal) | 0 |
| ITa-T15-005 | epoch 0: rows of the batch tensors whose targets under the mask do not decode to the trimmed answer + im_end (count) | 0 |

Measured with the prototype: 0, 0, 0, 0, 0 in 22 s.

## Stages and case IDs

- Stage 1 (task T15b1): `SplitMix64`, `StateOf`, `Shuffle` (written now,
  used in stage 2), `EncodeSample`, `HasWideEscape`, `GetStr`,
  `TTrainSet` (all of it); rows UT-TrainData-001 to 014.
- Stage 2 (task T15b2): `EpochOrder`, `EpochBatches`, `SkipBatches`,
  `BatchTensors`; rows UT-TrainData-101 to 110.
- ITa: ITa-T15-001 to 005.
