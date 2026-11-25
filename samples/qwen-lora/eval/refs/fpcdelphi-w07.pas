program GenericSort;
{$mode delphi}
uses
  SysUtils;
type
  TCompare<T> = function(const A, B: T): Integer;

  TSorter = class
    class procedure Sort<T>(var A: array of T; Compare: TCompare<T>);
  end;

class procedure TSorter.Sort<T>(var A: array of T; Compare: TCompare<T>);
var
  I, J: Integer;
  Tmp: T;
begin
  for I := High(A) downto 1 do
    for J := 0 to I - 1 do
      if Compare(A[J], A[J + 1]) > 0 then
      begin
        Tmp := A[J];
        A[J] := A[J + 1];
        A[J + 1] := Tmp;
      end;
end;

function IntAsc(const A, B: Integer): Integer;
begin
  Result := A - B;
end;

function StrDesc(const A, B: string): Integer;
begin
  Result := CompareStr(B, A);
end;

var
  Nums: array[0..4] of Integer = (5, 2, 9, 1, 7);
  Words: array[0..2] of string = ('pear', 'kiwi', 'apple');
  I: Integer;
begin
  TSorter.Sort<Integer>(Nums, IntAsc);
  TSorter.Sort<string>(Words, StrDesc);
  for I := 0 to High(Nums) do
  begin
    if I > 0 then
      Write(' ');
    Write(Nums[I]);
  end;
  WriteLn;
  for I := 0 to High(Words) do
  begin
    if I > 0 then
      Write(' ');
    Write(Words[I]);
  end;
  WriteLn;
end.
