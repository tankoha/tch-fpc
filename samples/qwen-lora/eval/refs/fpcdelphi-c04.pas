program Shapes2;
{$mode delphi}
type
  TKind = (Square, Circle, Triangle);
var
  K: TKind;
begin
  for K := Low(TKind) to High(TKind) do
    case K of
      Square: WriteLn('four corners');
      Circle: WriteLn('no corners');
    else
      WriteLn('three corners');
    end;
end.
