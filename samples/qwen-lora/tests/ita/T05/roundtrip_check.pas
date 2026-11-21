{ ITa driver of T05 (written by the harness owner; agents run it through
  tests/ita/T05-run.sh, they do not change it). Uses only the interfaces of
  GgufFile and QwenTokenizer.
    roundtrip_check MODEL CORPUS.jsonl
  For every record {"id": ..., "text": ...} prints id<TAB>OK when
  Decode(Encode(text)) = text byte for byte, else id<TAB>NG. }
program roundtrip_check;

{$mode objfpc}{$H+}

uses
  Classes, SysUtils, fpjson, jsonparser, GgufFile, QwenTokenizer;

var
  G: TGgufFile;
  T: TQwenTokenizer;
  S: TFileStream;
  Data, Line: RawByteString;
  J: TJSONData;
  Txt: UTF8String;
  P: Integer;
begin
  G := TGgufFile.Open(ParamStr(1));
  T := TQwenTokenizer.Create(G);
  S := TFileStream.Create(ParamStr(2), fmOpenRead or fmShareDenyNone);
  SetLength(Data, S.Size);
  if S.Size > 0 then S.ReadBuffer(Data[1], S.Size);
  S.Free;
  while Data <> '' do
  begin
    P := Pos(#10, Data);
    if P = 0 then P := Length(Data) + 1;
    Line := Copy(Data, 1, P - 1);
    Delete(Data, 1, P);
    if Line = '' then Continue;
    SetCodePage(Line, CP_UTF8, False);
    J := GetJSON(Line, False);  { measured: UseUTF8 = True corrupts non-ASCII in FPC 3.2.2 }
    Txt := TJSONObject(J).Strings['text'];
    if T.Decode(T.Encode(RawByteString(Txt))) = RawByteString(Txt) then
      WriteLn(TJSONObject(J).Get('id', ''), #9'OK')
    else
      WriteLn(TJSONObject(J).Get('id', ''), #9'NG');
    J.Free;
  end;
  T.Free;
  G.Free;
end.
