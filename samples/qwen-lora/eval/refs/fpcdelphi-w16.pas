program BigNumbers;
{$mode delphi}
var
  Q: QWord;
  F: Int64;
  I: Integer;
begin
  Q := High(QWord);
  WriteLn(Q);
  F := 1;
  for I := 1 to 20 do
  begin
    F := F * I;
    WriteLn(I, '! = ', F);
  end;
end.
