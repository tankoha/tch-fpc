program ParCount;
{$mode unleashed}
uses
  {$ifdef UNIX}cthreads,{$endif} SysUtils;

var
  By7, By11: Integer;

begin
  By7 := 0;
  By11 := 0;
  for parallel var I := 1 to 20000 do
  begin
    if I mod 7 = 0 then
      lock(By7) do By7 += 1;
    if I mod 11 = 0 then
      lock(By11) do By11 += 1;
  end;
  WriteLn($'by7 {By7}');
  WriteLn($'by11 {By11}');
end.
