program Hanoi;
{$mode objfpc}{$H+}
var
  Moves: Integer = 0;

procedure Solve(N: Integer; AFrom, ATo, AVia: Char);
begin
  if N = 0 then
    Exit;
  Solve(N - 1, AFrom, AVia, ATo);
  WriteLn('move disk ', N, ' from ', AFrom, ' to ', ATo);
  Inc(Moves);
  Solve(N - 1, AVia, ATo, AFrom);
end;

begin
  Solve(3, 'A', 'C', 'B');
  WriteLn('moves: ', Moves);
end.
