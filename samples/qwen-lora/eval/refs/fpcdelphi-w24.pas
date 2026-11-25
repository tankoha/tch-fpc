program Nim;
{$mode delphi}
var
  Sticks, Take: Integer;
  Player: Char;
begin
  Sticks := 15;
  Player := 'A';
  while True do
  begin
    if Player = 'A' then
    begin
      Take := Sticks mod 4;
      if Take = 0 then
        Take := 1;
    end
    else if Sticks >= 2 then
      Take := 2
    else
      Take := 1;
    Dec(Sticks, Take);
    WriteLn(Player, ' takes ', Take, ', ', Sticks, ' left');
    if Sticks = 0 then
      Break;
    if Player = 'A' then
      Player := 'B'
    else
      Player := 'A';
  end;
  WriteLn('winner: ', Player);
end.
