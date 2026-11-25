program ParallelSum;
{$mode unleashed}
uses
  {$ifdef UNIX}cthreads,{$endif} SysUtils;

var
  Total: Int64;

begin
  Total := 0;
  for parallel var I := 1 to 1000 do
    lock(Total) do Total += I;
  WriteLn($'total {Total}');
end.
