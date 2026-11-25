program SmallSet;
{$APPTYPE CONSOLE}
uses
  System.SysUtils;
type
  TSmallSet = record
    Bits: Cardinal;
    procedure Include(N: Integer);
    function Contains(N: Integer): Boolean;
    function Count: Integer;
  end;

procedure TSmallSet.Include(N: Integer);
begin
  Bits := Bits or (Cardinal(1) shl N);
end;

function TSmallSet.Contains(N: Integer): Boolean;
begin
  Result := (Bits and (Cardinal(1) shl N)) <> 0;
end;

function TSmallSet.Count: Integer;
begin
  Result := 0;
  for var I := 0 to 31 do
    if Contains(I) then
      Inc(Result);
end;

const
  Added: array[0..4] of Integer = (1, 3, 5, 7, 3);
var
  S: TSmallSet;
begin
  S.Bits := 0;
  for var N in Added do
    S.Include(N);
  Writeln('contains 3: ', BoolToStr(S.Contains(3), True));
  Writeln('contains 4: ', BoolToStr(S.Contains(4), True));
  Writeln('count ', S.Count);
end.
