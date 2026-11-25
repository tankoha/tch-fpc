program Life;
{$mode objfpc}{$H+}
const
  N = 6;
type
  TGrid = array[0..N - 1, 0..N - 1] of Boolean;

function Neighbours(const G: TGrid; R, C: Integer): Integer;
var
  DR, DC: Integer;
begin
  Result := 0;
  for DR := -1 to 1 do
    for DC := -1 to 1 do
      if ((DR <> 0) or (DC <> 0)) and (R + DR >= 0) and (R + DR < N) and
         (C + DC >= 0) and (C + DC < N) and G[R + DR, C + DC] then
        Inc(Result);
end;

procedure Step(var G: TGrid);
var
  Next: TGrid;
  R, C, K: Integer;
begin
  for R := 0 to N - 1 do
    for C := 0 to N - 1 do
    begin
      K := Neighbours(G, R, C);
      Next[R, C] := (K = 3) or (G[R, C] and (K = 2));
    end;
  G := Next;
end;

var
  G: TGrid;
  R, C, I: Integer;
begin
  for R := 0 to N - 1 do
    for C := 0 to N - 1 do
      G[R, C] := False;
  G[0, 1] := True;
  G[1, 2] := True;
  G[2, 0] := True;
  G[2, 1] := True;
  G[2, 2] := True;
  for I := 1 to 4 do
    Step(G);
  for R := 0 to N - 1 do
  begin
    for C := 0 to N - 1 do
      if G[R, C] then
        Write('#')
      else
        Write('.');
    WriteLn;
  end;
end.
