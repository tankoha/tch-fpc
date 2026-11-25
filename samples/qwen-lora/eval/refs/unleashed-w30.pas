program Durations;
{$mode unleashed}
uses
  SysUtils;

function Fmt(S: Integer): string;
begin
  var H := S div 3600;
  var M := (S mod 3600) div 60;
  var Sec := S mod 60;
  Result := (if H > 0 then $'{H}h ' else '') + (if M > 0 then $'{M}m ' else '') +
    (if Sec > 0 then $'{Sec}s' else '');
  Result := Result.Trim;
  if Result = '' then
    Result := '0s';
end;

begin
  for var S in [0, 59, 61, 3600, 3725, 86399] do
    WriteLn(Fmt(S));
end.
