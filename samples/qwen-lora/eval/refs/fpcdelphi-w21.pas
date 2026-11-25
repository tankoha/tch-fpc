program Rotate;
{$mode delphi}
type
  TM = array[0..2, 0..2] of Integer;

procedure Print(const M: TM);
var
  R, C: Integer;
begin
  for R := 0 to 2 do
  begin
    for C := 0 to 2 do
    begin
      if C > 0 then
        Write(' ');
      Write(M[R, C]);
    end;
    WriteLn;
  end;
end;

var
  M, Rot, Tr: TM;
  R, C: Integer;
begin
  for R := 0 to 2 do
    for C := 0 to 2 do
      M[R, C] := R * 3 + C + 1;
  for R := 0 to 2 do
    for C := 0 to 2 do
    begin
      Rot[C, 2 - R] := M[R, C];
      Tr[C, R] := M[R, C];
    end;
  Print(Rot);
  WriteLn('transpose');
  Print(Tr);
end.
