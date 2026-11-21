{ Test fixture of T05 (written by the harness owner; the agents use it, they
  do not change it): the tiny vocabulary of T05-facts "UT vocabulary" and a
  writer for a small GGUF file that carries it, so that the UT of
  QwenTokenizer and TokenizeCli run without a model file. }
unit TokenizerFixture;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, QwenTokenizer;

type
  TRawArray = array of RawByteString;
  TIntArray = array of Integer;

{ The UT vocabulary: ids 0..255 are ByteToUnicode(0..255) (type 1), then the
  tokens 256..273 and the 12 merges of the facts; EOS 267. }
procedure TinyVocab(out ATokens: TRawArray; out ATypes: TIntArray; out AMerges: TRawArray; out AEos: Integer);
{ TQwenTokenizer.CreateFromVocab on the UT vocabulary. }
function NewTinyTokenizer: TQwenTokenizer;
{ Writes a GGUF v3 file without tensors whose metadata is the UT vocabulary:
  tokenizer.ggml.model = AModel, tokenizer.ggml.pre = APre, tokens,
  token_type, merges, eos_token_id = 267. A key named in AOmitKey is left
  out (to test a missing key). }
procedure WriteTinyGguf(const APath: string; const AModel: RawByteString = 'gpt2';
  const APre: RawByteString = 'qwen35'; const AOmitKey: RawByteString = '');
{ Token ids as one line, separated by single spaces ('' for none). }
function IdsText(const A: array of Integer): string;
{ Words as one line: each word between `|`, for example |hello|| world| }
function WordsText(const A: array of RawByteString): RawByteString;
{ Writes AText to APath byte for byte. }
procedure WriteBytes(const APath: string; const AText: RawByteString);

implementation

procedure TinyVocab(out ATokens: TRawArray; out ATypes: TIntArray; out AMerges: TRawArray; out AEos: Integer);
const
  Extra: array[0..17] of RawByteString = ('he', 'll', 'hell', 'hello', #1'w', 'or', #1'wor', 'ld',
    #1'world', 'aa', '<|im_start|>', '<|im_end|>', '<think>', '<tts_text_bos>', '<tts_text_bos_single>',
    'ab', 'bc', '<|unused|>');
  ExtraTypes: array[0..17] of Integer = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 4, 3, 3, 1, 1, 5);
  Merges: array[0..11] of RawByteString = ('h e', 'l l', 'he ll', 'hell o', #1' w', 'o r',
    #1'w or', 'l d', #1'wor ld', 'a a', 'b c', 'a b');
var
  I: Integer;
  Sp: RawByteString;
begin
  Sp := ByteToUnicode(32);   { #1 above stands for the encoded space }
  SetLength(ATokens, 274); SetLength(ATypes, 274);
  for I := 0 to 255 do
  begin
    ATokens[I] := ByteToUnicode(I); ATypes[I] := 1;
  end;
  for I := 0 to 17 do
  begin
    if Extra[I][1] = #1 then
      ATokens[256 + I] := Sp + Copy(Extra[I], 2, Length(Extra[I]) - 1)
    else
      ATokens[256 + I] := Extra[I];
    ATypes[256 + I] := ExtraTypes[I];
  end;
  SetLength(AMerges, 12);
  for I := 0 to 11 do
    if Merges[I][1] = #1 then
      AMerges[I] := Sp + Copy(Merges[I], 2, Length(Merges[I]) - 1)
    else
      AMerges[I] := Merges[I];
  AEos := 267;
end;

function NewTinyTokenizer: TQwenTokenizer;
var
  T, M: TRawArray;
  Ty: TIntArray;
  E: Integer;
begin
  TinyVocab(T, Ty, M, E);
  Result := TQwenTokenizer.CreateFromVocab(T, Ty, M, E);
end;

procedure WriteTinyGguf(const APath: string; const AModel: RawByteString;
  const APre: RawByteString; const AOmitKey: RawByteString);
var
  S: TFileStream;
  T, M: TRawArray;
  Ty: TIntArray;
  E, I, NKV: Integer;
  Z: Byte;

  procedure W4(V: Cardinal); begin S.WriteBuffer(V, 4); end;
  procedure W8(V: QWord); begin S.WriteBuffer(V, 8); end;
  procedure WS(const A: RawByteString);
  begin
    W8(Length(A));
    if A <> '' then S.WriteBuffer(A[1], Length(A));
  end;
  function Want(const AKey: RawByteString): Boolean;
  begin
    Result := AKey <> AOmitKey;
    if Result then WS(AKey);
  end;

begin
  TinyVocab(T, Ty, M, E);
  NKV := 6;
  if AOmitKey <> '' then Dec(NKV);
  S := TFileStream.Create(APath, fmCreate);
  try
    S.WriteBuffer(PAnsiChar('GGUF')^, 4);
    W4(3); W8(0); W8(NKV);
    if Want('tokenizer.ggml.model') then begin W4(8); WS(AModel); end;
    if Want('tokenizer.ggml.pre') then begin W4(8); WS(APre); end;
    if Want('tokenizer.ggml.tokens') then
    begin
      W4(9); W4(8); W8(Length(T));
      for I := 0 to High(T) do WS(T[I]);
    end;
    if Want('tokenizer.ggml.token_type') then
    begin
      W4(9); W4(5); W8(Length(Ty));
      for I := 0 to High(Ty) do W4(Cardinal(Ty[I]));
    end;
    if Want('tokenizer.ggml.merges') then
    begin
      W4(9); W4(8); W8(Length(M));
      for I := 0 to High(M) do WS(M[I]);
    end;
    if Want('tokenizer.ggml.eos_token_id') then begin W4(4); W4(E); end;
    Z := 0;
    while S.Position mod 32 <> 0 do S.WriteBuffer(Z, 1);
  finally
    S.Free;
  end;
end;

function IdsText(const A: array of Integer): string;
var
  I: Integer;
begin
  Result := '';
  for I := 0 to High(A) do
  begin
    if I > 0 then Result := Result + ' ';
    Result := Result + IntToStr(A[I]);
  end;
end;

function WordsText(const A: array of RawByteString): RawByteString;
var
  I: Integer;
begin
  Result := '|';
  for I := 0 to High(A) do
    Result := Result + A[I] + '|';
end;

procedure WriteBytes(const APath: string; const AText: RawByteString);
var
  S: TFileStream;
begin
  S := TFileStream.Create(APath, fmCreate);
  try
    if AText <> '' then S.WriteBuffer(AText[1], Length(AText));
  finally
    S.Free;
  end;
end;

end.
