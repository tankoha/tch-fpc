program Collatz;
{$mode unleashed}
uses
  SysUtils;

begin
  var N: Int64 := 27;
  var Steps := 0;
  var Peak := N;
  var First := '';
  var Shown := 0;
  while True do
  begin
    if Shown < 10 then
    begin
      First += (if Shown > 0 then ' ' else '') + IntToStr(N);
      Inc(Shown);
    end;
    if N = 1 then
      Break;
    N := if Odd(N) then 3 * N + 1 else N div 2;
    Inc(Steps);
    if N > Peak then
      Peak := N;
  end;
  WriteLn($'steps {Steps}');
  WriteLn($'peak {Peak}');
  WriteLn(First);
end.
