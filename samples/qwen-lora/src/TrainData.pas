{$mode objfpc}{$H+}
{ Training data of a LoRA run (DESIGN 8.14 "Batches", "Loss mask", 8.16; T15).
  Records:
    A data set file is JSONL, one record per line with the string keys id,
    dialect, prompt, answer and class (the verification class of 8.16);
    other keys are ignored. Text is raw UTF-8: fpjson turns a \u escape of a
    non-ASCII character into '?', so such an escape is refused.
  Samples:
    A record becomes the conversation user = prompt, assistant = answer,
    rendered without thinking (QwenChat, EnableThinking False: the empty
    think block before the answer) and tokenised; the loss covers the
    answer and its closing im_end (EncodeTraining's [LossFrom, LossTo)).
  Batches:
    An epoch's order is a permutation of the samples from (seed, epoch); it
    is cut into pools, each pool sorted by length and cut into batches of
    at most Budget positions (count times the longest), and the batches of a
    pool are shuffled. Everything is a function of (seed, epoch), so a run
    resumes by skipping the batches of the samples already consumed.
  Interface by the harness owner (tasks/T15-facts.md); the bodies are
  written in T15b. }
unit TrainData;

interface

uses
  SysUtils, Classes, Math, fpjson, jsonparser, Tch, QwenTokenizer, QwenChat;

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

end.
