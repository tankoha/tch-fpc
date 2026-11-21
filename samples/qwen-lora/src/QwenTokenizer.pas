{ QwenTokenizer — the byte-level BPE tokenizer of the Qwen3.5/3.8 GGUF
  (tokenizer.ggml.model gpt2, pre qwen35), per DESIGN.md 8.10 and
  tasks/T05-facts.md (authoritative). Encode equals the bundled
  llama-server's /tokenize (special tokens parsed, no BOS/EOS); Decode
  gives the bytes back.
  Skeleton by the harness owner (the facts' interface); stage 1 (T05b1)
  writes ByteToUnicode, CreateFromVocab/Init, PreTokenize and the
  properties, stage 2 (T05b2) EncodeWord, Encode, Decode, SpecialId and
  Create. }
unit QwenTokenizer;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Generics.Collections, GgufFile;

type
  { Every error of this unit. }
  ETokenizerError = class(Exception);
  TTokenArray = array of Integer;
  TWordArray = array of RawByteString;
  TTokDict = specialize TDictionary<RawByteString, Integer>;

  TQwenTokenizer = class
  private
    FTokens: array of RawByteString;   { vocabulary strings, byte-encoded (facts "Byte map") }
    FTypes: array of Integer;          { token types: 1 normal, 3 control, 4 user-defined, 5 unused }
    FEosId: Integer;
    FTokenId: TTokDict;                { vocabulary string -> first id }
    FMergeRank: TTokDict;              { merge string "left right" -> rank (index) }
    FSpecials: array of Integer;       { ids of types 3 and 4, longest text first }
    FByteDec: TTokDict;                { ByteToUnicode(b) -> b }
    procedure Init(const ATokens: array of RawByteString; const ATypes: array of Integer;
      const AMerges: array of RawByteString; AEosId: Integer);
    procedure EncodeWord(const AWord: RawByteString; var AOut: TTokenArray; var ACount: Integer);
    function GetVocabSize: Integer;
  public
    constructor Create(AGguf: TGgufFile);
    constructor CreateFromVocab(const ATokens: array of RawByteString; const ATypes: array of Integer;
      const AMerges: array of RawByteString; AEosId: Integer);
    destructor Destroy; override;
    function PreTokenize(const AText: RawByteString): TWordArray;
    function Encode(const AText: RawByteString): TTokenArray;
    function Decode(const ATokens: array of Integer): RawByteString;
    function SpecialId(const AText: RawByteString): Integer;
    property VocabSize: Integer read GetVocabSize;
    property EosId: Integer read FEosId;
  end;

function ByteToUnicode(AByte: Byte): RawByteString;

implementation

uses
  unicodedata;
function CodepointToUTF8(cp: Word): RawByteString; forward;
var
  ByteToUnicodeMap: array of RawByteString;

function ByteToUnicode(AByte: Byte): RawByteString;
var
  cp: Word;
  b: Byte;
begin
  // Map byte to Unicode code point per facts.
  // Bytes in the printable ranges map to themselves.
  if ((AByte >= $21) and (AByte <= $7E)) or
     ((AByte >= $A1) and (AByte <= $AC)) or
     ((AByte >= $AE) and (AByte <= $FF)) then
    cp := AByte
  else
  begin
    // Non‑standard bytes get sequential code points starting at $0100.
    // Compute their index in the ordered list of such bytes.
    // The order is increasing byte value.
    // Pre‑compute a static table at first call.
    if Length(ByteToUnicodeMap)=0 then
    begin
      SetLength(ByteToUnicodeMap, 256);
      for b := 0 to 255 do
      begin
        if ((b >= $21) and (b <= $7E)) or
           ((b >= $A1) and (b <= $AC)) or
           ((b >= $AE) and (b <= $FF)) then
          ByteToUnicodeMap[b] := Chr(b)
        else
          ByteToUnicodeMap[b] := '';
      end;
      // assign sequential code points starting at $0100 to empty slots
      cp := $0100;
      for b := 0 to 255 do
        if ByteToUnicodeMap[b] = '' then
        begin
          ByteToUnicodeMap[b] := CodepointToUTF8(cp);
          Inc(cp);
        end;
    end;
    Result := ByteToUnicodeMap[AByte];
    Exit;
  end;
  // Encode code point cp as UTF‑8 (cp fits in two bytes for our range).
  Result := CodepointToUTF8(cp);
end;

// Helper to encode a Unicode code point (<= $07FF) to UTF‑8.
function CodepointToUTF8(cp: Word): RawByteString;
begin
  if cp < $80 then
    Result := Chr(cp)
  else
    Result := Chr(($C0 or (cp shr 6))) + Chr(($80 or (cp and $3F)));
end;


// Character class helpers (based on unicodedata constants)
type
  TCharClass = record
    letter: Boolean;
    mark: Boolean;
    number: Boolean;
    space: Boolean;
    defined: Boolean;
  end;

function IsSpaceCp(cp: Cardinal): Boolean;
begin
  // List of Unicode code points that count as spaces (per facts).
  case cp of
    $09..$0D, $20, $85, $A0, $1680,
    $2000..$200A, $2028, $2029, $202F, $205F, $3000:
      Result := True;
  else
    Result := False;
  end;
end;

function CharClassOf(cp: Cardinal): TCharClass;
var
  cat: Integer;
begin
  // Get Unicode category from unicodedata.
  cat := GetProps(cp)^.Category;
  Result.letter := (cat >= UGC_UppercaseLetter) and (cat <= UGC_OtherLetter);
  Result.mark := (cat >= UGC_NonSpacingMark) and (cat <= UGC_EnclosingMark);
  Result.number := (cat >= UGC_DecimalNumber) and (cat <= UGC_OtherNumber);
  Result.space := IsSpaceCp(cp);
  Result.defined := (cat <> UGC_Unassigned) or Result.space;
end;

function LowerCp(cp: Cardinal): Cardinal;
begin
  if (cp >= Ord('A')) and (cp <= Ord('Z')) then
    Result := cp + (Ord('a') - Ord('A'))
  else
    Result := cp;
end;

constructor TQwenTokenizer.Create(AGguf: TGgufFile);
var
  tokIdx, typIdx, mrgIdx, eosIdx, modelIdx, preIdx, ii: Integer; // added modelIdx, preIdx
  modelStr, preStr: RawByteString;
  ATokens: array of RawByteString;
  ATypes: array of Integer;
  AMerges: array of RawByteString;
  AEosId: Integer;
begin
  inherited Create;
  // Load required keys from the GGUF file
  tokIdx := AGguf.FindKey('tokenizer.ggml.tokens');
  if tokIdx < 0 then
    raise ETokenizerError.Create('tokenizer: key tokenizer.ggml.tokens missing');
  typIdx := AGguf.FindKey('tokenizer.ggml.token_type');
  if typIdx < 0 then
    raise ETokenizerError.Create('tokenizer: key tokenizer.ggml.token_type missing');
  mrgIdx := AGguf.FindKey('tokenizer.ggml.merges');
  if mrgIdx < 0 then
    raise ETokenizerError.Create('tokenizer: key tokenizer.ggml.merges missing');
  eosIdx := AGguf.FindKey('tokenizer.ggml.eos_token_id');
  if eosIdx < 0 then
    raise ETokenizerError.Create('tokenizer: key tokenizer.ggml.eos_token_id missing');
  modelIdx := AGguf.FindKey('tokenizer.ggml.model');
  if modelIdx < 0 then
    raise ETokenizerError.Create('tokenizer: key tokenizer.ggml.model missing');
  modelStr := AGguf.KV[modelIdx].StrVal;
  if modelStr <> 'gpt2' then
    raise ETokenizerError.CreateFmt('tokenizer: model %s, expected gpt2', [modelStr]);
  preIdx := AGguf.FindKey('tokenizer.ggml.pre');
  if preIdx < 0 then
    raise ETokenizerError.Create('tokenizer: key tokenizer.ggml.pre missing');
  preStr := AGguf.KV[preIdx].StrVal;
  if preStr <> 'qwen35' then
    raise ETokenizerError.CreateFmt('tokenizer: pre %s, expected qwen35', [preStr]);
  // Extract arrays
  ATokens := AGguf.KV[tokIdx].Strs;
  // Convert token types to Integer array
  SetLength(ATypes, Length(AGguf.KV[typIdx].Ints));
  for ii := 0 to High(ATypes) do
    ATypes[ii] := Integer(AGguf.KV[typIdx].Ints[ii]);
  AMerges := AGguf.KV[mrgIdx].Strs;
  AEosId := Integer(AGguf.KV[eosIdx].IntVal);
  Init(ATokens, ATypes, AMerges, AEosId);
end;

constructor TQwenTokenizer.CreateFromVocab(const ATokens: array of RawByteString;
  const ATypes: array of Integer; const AMerges: array of RawByteString; AEosId: Integer);
begin
  inherited Create;
  Init(ATokens, ATypes, AMerges, AEosId);
end;

procedure TQwenTokenizer.Init(const ATokens: array of RawByteString; const ATypes: array of Integer;
  const AMerges: array of RawByteString; AEosId: Integer);
var
  I, j, tmpIdx: Integer;
  tokenStr: RawByteString;
begin
  // Verify token and type counts
  if Length(ATokens) <> Length(ATypes) then
    raise ETokenizerError.CreateFmt('tokenizer: %d tokens but %d types', [Length(ATokens), Length(ATypes)]);
  // Verify EOS id range
  if (AEosId < 0) or (AEosId >= Length(ATokens)) then
    raise ETokenizerError.CreateFmt('tokenizer: eos id %d out of range', [AEosId]);
  // Store vocab and EOS
  FEosId := AEosId;
  SetLength(FTokens, Length(ATokens));
  SetLength(FTypes, Length(ATypes));
  for I := 0 to High(ATokens) do
  begin
    FTokens[I] := ATokens[I];
    FTypes[I] := ATypes[I];
  end;
  // Dictionaries
  FTokenId := TTokDict.Create;
  for I := 0 to High(FTokens) do
    if not FTokenId.ContainsKey(FTokens[I]) then
      FTokenId.Add(FTokens[I], I);

  FMergeRank := TTokDict.Create;
  for I := 0 to High(AMerges) do
    if not FMergeRank.ContainsKey(AMerges[I]) then
      FMergeRank.Add(AMerges[I], I);

  // Verify byte map tokens exist and build inverse map
  FByteDec := TTokDict.Create;
  for I := 0 to 255 do
  begin
    tokenStr := ByteToUnicode(I);
    if not FTokenId.ContainsKey(tokenStr) then
      raise ETokenizerError.CreateFmt('tokenizer: no token for byte %d', [I]);
    FByteDec.Add(tokenStr, I);
  end;

  // Build specials list (ids of types 3 and 4) – sorted longest text first, then lower id
  SetLength(FSpecials, 0);
  for I := 0 to High(FTypes) do
    if (FTypes[I] = 3) or (FTypes[I] = 4) then
    begin
      SetLength(FSpecials, Length(FSpecials) + 1);
      FSpecials[High(FSpecials)] := I;
    end;
  // Sorting of specials will be performed dynamically during Encode
end;

destructor TQwenTokenizer.Destroy;
begin
  FTokenId.Free;
  FMergeRank.Free;
  FByteDec.Free;
  inherited Destroy;
end;

function TQwenTokenizer.GetVocabSize: Integer;
begin
  Result := Length(FTokens);
end;

procedure TQwenTokenizer.EncodeWord(const AWord: RawByteString; var AOut: TTokenArray; var ACount: Integer);
var
  Sym: array of RawByteString;
  RankArr: array of Integer;
  N, i, best, bi: Integer;
  function GetRank(const left, right: RawByteString): Integer;
  var
    mergeStr: RawByteString;
    rankVal: Integer;
  begin
    mergeStr := left + ' ' + right;
    if FMergeRank.TryGetValue(mergeStr, rankVal) then
      Result := rankVal
    else
      Result := MaxInt;
  end;
begin
  N := Length(AWord);
  SetLength(Sym, N);
  SetLength(RankArr, N);
  // Convert each byte to its Unicode symbol
  for i := 1 to N do
    Sym[i-1] := ByteToUnicode(Byte(Ord(AWord[i])));
  // Initial ranks
  for i := 0 to N-2 do
    RankArr[i] := GetRank(Sym[i], Sym[i+1]);
  if N > 0 then
    RankArr[N-1] := MaxInt;
  // BPE merge loop
  while N > 1 do
  begin
    best := MaxInt;
    bi := -1;
    for i := 0 to N-2 do
      if RankArr[i] < best then
      begin
        best := RankArr[i];
        bi := i;
      end;
    if bi < 0 then Break;
    // merge
    Sym[bi] := Sym[bi] + Sym[bi+1];
    // shift left
    for i := bi+1 to N-2 do
    begin
      Sym[i] := Sym[i+1];
      RankArr[i] := RankArr[i+1];
    end;
    Dec(N);
    SetLength(Sym, N);
    SetLength(RankArr, N);
    if N > 0 then
      RankArr[N-1] := MaxInt;
    // recompute affected ranks
    if bi > 0 then
      RankArr[bi-1] := GetRank(Sym[bi-1], Sym[bi]);
    if bi < N-1 then
      RankArr[bi] := GetRank(Sym[bi], Sym[bi+1]);
  end;
  // Output token ids for remaining symbols
  for i := 0 to N-1 do
  begin
    SetLength(AOut, ACount + 1);
    AOut[ACount] := FTokenId[Sym[i]];
    Inc(ACount);
  end;
end;

function TQwenTokenizer.PreTokenize(const AText: RawByteString): TWordArray;
var
  Cps: array of Cardinal;          // Unicode code points
  BytePos: array of Integer;       // start byte index for each code point (1‑based)
  N, PosIdx, PrevIdx: Integer;
  ResultList: TWordArray;

  f, nextF, f2, curF: TCharClass;
  c1, c2: Cardinal;
  nw, lastRN: Integer;

  procedure AddTok(eIdx: Integer);
  var
    startByte, lenBytes: Integer;
    word: RawByteString;
  begin
    if eIdx > PrevIdx then
    begin
      startByte := BytePos[PrevIdx];
      lenBytes := BytePos[eIdx] - startByte;
      word := Copy(AText, startByte, lenBytes);
      SetLength(ResultList, Length(ResultList) + 1);
      ResultList[High(ResultList)] := word;
    end;
    PrevIdx := eIdx;
  end;

  function GetCharClass(idx: Integer): TCharClass;
  begin
    if (idx >= 0) and (idx < N) then
      Result := CharClassOf(Cps[idx])
    else
    begin
      Result.letter := False; Result.mark := False; Result.number := False;
      Result.space := False; Result.defined := False;
    end;
  end;

  function CpAt(idx: Integer): Cardinal;
  begin
    if (idx >= 0) and (idx < N) then
      Result := Cps[idx]
    else
      Result := $FFFFFFFF; // NoCp
  end;

var
  iByte, b, bytesNeeded, k: Integer;
  cp: Cardinal;
begin
  // Decode UTF‑8, detecting errors.
  N := 0;
  SetLength(Cps, 0);
  SetLength(BytePos, 0);
  iByte := 1;
  while iByte <= Length(AText) do
  begin
    b := Ord(AText[iByte]);
    if b < $80 then begin cp := b; bytesNeeded := 1; end
    else if (b >= $C2) and (b <= $DF) then begin cp := b and $1F; bytesNeeded := 2; end
    else if (b >= $E0) and (b <= $EF) then begin cp := b and $0F; bytesNeeded := 3; end
    else if (b >= $F0) and (b <= $F4) then begin cp := b and $07; bytesNeeded := 4; end
    else raise ETokenizerError.CreateFmt('invalid UTF-8 at byte %d', [iByte]);
    if iByte + bytesNeeded - 1 > Length(AText) then
      raise ETokenizerError.CreateFmt('invalid UTF-8 at byte %d', [iByte]);
    for k := 1 to bytesNeeded - 1 do
    begin
      if (Ord(AText[iByte + k]) and $C0) <> $80 then
        raise ETokenizerError.CreateFmt('invalid UTF-8 at byte %d', [iByte + k]);
      cp := (cp shl 6) or (Ord(AText[iByte + k]) and $3F);
    end;
    SetLength(Cps, N + 1);
    SetLength(BytePos, N + 1);
    Cps[N] := cp;
    BytePos[N] := iByte;
    Inc(N);
    Inc(iByte, bytesNeeded);
  end;
  // sentinel for end
  SetLength(BytePos, N+1);
  BytePos[N] := Length(AText) + 1;

  // scanning
  SetLength(ResultList, 0);
  PosIdx := 0;
  PrevIdx := 0;

  while PosIdx < N do
  begin
    cp := CpAt(PosIdx);
    f := GetCharClass(PosIdx);
    // contractions
    if (cp = Ord('''')) and (PosIdx + 1 < N) then
    begin
      c1 := LowerCp(CpAt(PosIdx+1));
      if c1 in [Ord('s'), Ord('t'), Ord('m'), Ord('d')] then
      begin
        AddTok(PosIdx+2);
        PosIdx := PosIdx + 2;
        Continue;
      end;
      if (PosIdx + 2 < N) then
      begin
        c2 := LowerCp(CpAt(PosIdx+2));
        if ((c1 = Ord('r')) and (c2 = Ord('e'))) or
           ((c1 = Ord('v')) and (c2 = Ord('e'))) or
           ((c1 = Ord('l')) and (c2 = Ord('l'))) then
        begin
          AddTok(PosIdx+3);
          PosIdx := PosIdx + 3;
          Continue;
        end;
      end;
    end;
    // word letters/marks pattern
    if not ((cp = $0D) or (cp = $0A) or f.number) then
    begin
      nextF := GetCharClass(PosIdx+1);
      if f.letter or f.mark or nextF.letter or nextF.mark then
      begin
        Inc(PosIdx);
        while (PosIdx < N) and (GetCharClass(PosIdx).letter or GetCharClass(PosIdx).mark) do
          Inc(PosIdx);
        AddTok(PosIdx);
        Continue;
      end;
    end;
    // numbers
    if f.number then
    begin
      Inc(PosIdx);
      AddTok(PosIdx);
      Continue;
    end;
    // non-space punctuation
    if cp = Ord(' ') then
      f2 := GetCharClass(PosIdx+1)
    else
      f2 := f;
    if not (f2.space or f2.letter or f2.mark or f2.number) and f.defined then
    begin
      if cp = Ord(' ') then
        Inc(PosIdx);
      while (PosIdx < N) do
      begin
        curF := GetCharClass(PosIdx);
        if curF.space or curF.letter or curF.mark or curF.number then Break;
        if not curF.defined then Break;
        Inc(PosIdx);
      end;
      while (PosIdx < N) and (CpAt(PosIdx) in [$0D, $0A]) do Inc(PosIdx);
      AddTok(PosIdx);
      Continue;
    end;
    // spaces handling
    nw := 0; lastRN := 0;
    while (PosIdx + nw < N) and GetCharClass(PosIdx+nw).space do
    begin
      if CpAt(PosIdx+nw) in [$0D, $0A] then
        lastRN := PosIdx + nw + 1;
      Inc(nw);
    end;
    if lastRN > 0 then
    begin
      PosIdx := lastRN;
      AddTok(PosIdx);
      Continue;
    end;
    if (nw > 1) and (CpAt(PosIdx+nw) <> $FFFFFFFF) then
    begin
      PosIdx := PosIdx + nw - 1;
      AddTok(PosIdx);
      Continue;
    end;
    if nw > 0 then
    begin
      PosIdx := PosIdx + nw;
      AddTok(PosIdx);
      Continue;
    end;
    // anything else alone
    Inc(PosIdx);
    AddTok(PosIdx);
  end;
  Result := ResultList;
end;

function TQwenTokenizer.Encode(const AText: RawByteString): TTokenArray;
var
  fragments: array of RawByteString;
  i, j, fragIdx, outCount, tmp: Integer;
  wordList: TWordArray;
  tokenArray: TTokenArray;
  SortedSpecials: array of Integer;
  word: RawByteString;
  specStr: RawByteString;
  newFrags: array of RawByteString;
  src: RawByteString;
  curPos: Integer;
  p: Integer;
  sid: Integer;

begin
  // Step 1: split specials first
  SetLength(fragments, 1);
  fragments[0] := AText;
  // Prepare sorted specials (longest token first, then lower id)
  SetLength(SortedSpecials, Length(FSpecials));
  for i := 0 to High(FSpecials) do
    SortedSpecials[i] := FSpecials[i];
  // simple bubble sort
  for i := 0 to High(SortedSpecials) - 1 do
    for j := i + 1 to High(SortedSpecials) do
      if (Length(FTokens[SortedSpecials[i]]) < Length(FTokens[SortedSpecials[j]])) or
         ((Length(FTokens[SortedSpecials[i]]) = Length(FTokens[SortedSpecials[j]])) and (SortedSpecials[i] > SortedSpecials[j])) then
      begin
        tmp := SortedSpecials[i];
        SortedSpecials[i] := SortedSpecials[j];
        SortedSpecials[j] := tmp;
      end;

  for i := 0 to High(SortedSpecials) do
  begin
    specStr := FTokens[SortedSpecials[i]];
    SetLength(newFrags, 0);
    for fragIdx := 0 to High(fragments) do
    begin
      // If fragment is already a special token, keep it
      if SpecialId(fragments[fragIdx]) >= 0 then
      begin
        SetLength(newFrags, Length(newFrags) + 1);
        newFrags[High(newFrags)] := fragments[fragIdx];
        Continue;
      end;
      src := fragments[fragIdx];
      curPos := 1;
      while curPos <= Length(src) do
      begin
        p := Pos(specStr, Copy(src, curPos, MaxInt));
        if p > 0 then
        begin
          p := curPos + p - 1;
          // raw before
          if p > curPos then
          begin
            SetLength(newFrags, Length(newFrags) + 1);
            newFrags[High(newFrags)] := Copy(src, curPos, p - curPos);
          end;
          // special token
          SetLength(newFrags, Length(newFrags) + 1);
          newFrags[High(newFrags)] := specStr;
          curPos := p + Length(specStr);
        end
        else
        begin
          // remaining raw
          SetLength(newFrags, Length(newFrags) + 1);
          newFrags[High(newFrags)] := Copy(src, curPos, Length(src) - curPos + 1);
          Break;
        end;
      end;
    end;
    fragments := newFrags;
  end;

  // Step 2: encode each fragment
  tokenArray := nil;
  for fragIdx := 0 to High(fragments) do
  begin
    sid := SpecialId(fragments[fragIdx]);
    if sid >= 0 then
    begin
      SetLength(tokenArray, Length(tokenArray) + 1);
      tokenArray[High(tokenArray)] := sid;
    end
    else
    begin
      wordList := PreTokenize(fragments[fragIdx]);
      for i := 0 to High(wordList) do
      begin
        outCount := Length(tokenArray);
        EncodeWord(wordList[i], tokenArray, outCount);
        SetLength(tokenArray, outCount);
      end;
    end;
  end;
  Result := tokenArray;
end;



function TQwenTokenizer.Decode(const ATokens: array of Integer): RawByteString;
var
  i, pos, n, b, v: Integer;
  tokenStr: RawByteString;
begin
  Result := '';
  for i := 0 to High(ATokens) do
  begin
    if (ATokens[i] < 0) or (ATokens[i] >= Length(FTokens)) then
      raise ETokenizerError.CreateFmt('token id %d out of range', [ATokens[i]]);
    tokenStr := FTokens[ATokens[i]];
    if (FTypes[ATokens[i]] = 3) or (FTypes[ATokens[i]] = 4) then
    begin
      Result := Result + tokenStr;
      Continue;
    end;
    pos := 1;
    while pos <= Length(tokenStr) do
    begin
      b := Ord(tokenStr[pos]);
      if b < $80 then n := 1
      else if b < $E0 then n := 2
      else if b < $F0 then n := 3
      else n := 4;
      if not FByteDec.TryGetValue(Copy(tokenStr, pos, n), v) then
        raise ETokenizerError.CreateFmt('token %d: character not in the byte map', [ATokens[i]]);
      Result := Result + Chr(v);
      Inc(pos, n);
    end;
  end;
end;
function TQwenTokenizer.SpecialId(const AText: RawByteString): Integer;
var
  id: Integer;
begin
  if FTokenId.TryGetValue(AText, id) then
  begin
    if (FTypes[id] = 3) or (FTypes[id] = 4) then
      Result := id
    else
      Result := -1;
  end
  else
    Result := -1;
end;

end.
