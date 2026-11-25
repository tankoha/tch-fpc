program Knapsack;
{$mode delphi}
const
  W: array[1..4] of Integer = (5, 4, 6, 3);
  V: array[1..4] of Integer = (10, 40, 30, 50);
  Cap = 10;
var
  Best: array[0..4, 0..Cap] of Integer;
  I, C: Integer;
  Chosen: array of Integer;
begin
  for C := 0 to Cap do
    Best[0, C] := 0;
  for I := 1 to 4 do
    for C := 0 to Cap do
    begin
      Best[I, C] := Best[I - 1, C];
      if (W[I] <= C) and (Best[I - 1, C - W[I]] + V[I] > Best[I, C]) then
        Best[I, C] := Best[I - 1, C - W[I]] + V[I];
    end;
  WriteLn('best ', Best[4, Cap]);
  C := Cap;
  Chosen := nil;
  for I := 4 downto 1 do
    if Best[I, C] <> Best[I - 1, C] then
    begin
      Insert(I, Chosen, 0);
      Dec(C, W[I]);
    end;
  for I := 0 to High(Chosen) do
  begin
    if I > 0 then
      Write(' ');
    Write(Chosen[I]);
  end;
  WriteLn;
end.
