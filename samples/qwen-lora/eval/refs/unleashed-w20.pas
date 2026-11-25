program Rings;
{$mode unleashed}
uses
  SysUtils;
type
  TRing<T> = class
  private
    FData: array[0..2] of T;
    FStart, FCount: Integer;
  public
    procedure Add(const V: T);
    function Items: TArray<T>;
  end;

procedure TRing<T>.Add(const V: T);
begin
  if FCount < 3 then
  begin
    FData[(FStart + FCount) mod 3] := V;
    Inc(FCount);
  end
  else
  begin
    FData[FStart] := V;
    FStart := (FStart + 1) mod 3;
  end;
end;

function TRing<T>.Items: TArray<T>;
begin
  SetLength(Result, FCount);
  for var I := 0 to FCount - 1 do
    Result[I] := FData[(FStart + I) mod 3];
end;

begin
  var R := autofree TRing<Integer>.Create;
  for var I := 1 to 5 do
    R.Add(I);
  var Line := '';
  for var V in R.Items do
    Line += IntToStr(V) + ' ';
  WriteLn(Line.Trim);
  var S := autofree TRing<string>.Create;
  for var C in ['a', 'b', 'c', 'd'] do
    S.Add(C);
  WriteLn(string.Join(' ', S.Items));
end.
