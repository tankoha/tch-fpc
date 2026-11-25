program Transforms;
{$mode delphi}
uses
  SysUtils;
type
  TTransform = function(const S: string): string;

function Upper(const S: string): string;
begin
  Result := UpperCase(S);
end;

function Reverse(const S: string): string;
var
  I: Integer;
begin
  Result := '';
  for I := Length(S) downto 1 do
    Result := Result + S[I];
end;

function Bracket(const S: string): string;
begin
  Result := '[' + S + ']';
end;

const
  Steps: array[0..2] of TTransform = (Upper, Reverse, Bracket);
var
  S: string;
  I: Integer;
begin
  S := 'pascal';
  for I := 0 to High(Steps) do
  begin
    S := Steps[I](S);
    WriteLn(S);
  end;
end.
