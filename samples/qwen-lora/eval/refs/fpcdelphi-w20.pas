program ParseAges;
{$mode delphi}
uses
  SysUtils;
type
  EParseError = class(Exception);
  EEmptyInput = class(EParseError);
  EBadNumber = class(EParseError);

function ParseAge(const S: string): Integer;
begin
  if S = '' then
    raise EEmptyInput.Create('no input');
  if not TryStrToInt(S, Result) then
    raise EBadNumber.CreateFmt('"%s" is not a number', [S]);
  if (Result < 0) or (Result > 150) then
    raise EParseError.CreateFmt('%d is out of range', [Result]);
end;

procedure Try1(const S: string);
var
  N: Integer;
begin
  try
    N := ParseAge(S);
    WriteLn('ok ', N);
  except
    on E: EParseError do
      WriteLn(E.ClassName, ': ', E.Message);
  end;
end;

begin
  Try1('');
  Try1('abc');
  Try1('200');
  Try1('42');
end.
