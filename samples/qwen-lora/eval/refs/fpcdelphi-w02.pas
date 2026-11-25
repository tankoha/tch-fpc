program Pairs;
{$mode delphi}
uses
  SysUtils;
type
  TPair<TKey, TValue> = record
    Key: TKey;
    Value: TValue;
  end;
  TNamePair = TPair<string, Integer>;

function MakePair(const AKey: string; AValue: Integer): TNamePair;
begin
  Result.Key := AKey;
  Result.Value := AValue;
end;

var
  P: array[0..2] of TNamePair;
  I: Integer;
begin
  P[0] := MakePair('one', 1);
  P[1] := MakePair('two', 2);
  P[2] := MakePair('three', 3);
  for I := 0 to High(P) do
    WriteLn(P[I].Key, '=', P[I].Value);
end.
