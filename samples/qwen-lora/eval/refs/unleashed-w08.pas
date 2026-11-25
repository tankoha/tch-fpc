program SafeParse;
{$mode unleashed}
uses
  SysUtils;

begin
  for var S in ['42', 'q7', '-3', ''] do
  begin
    var N := try StrToInt(S) except on E: EConvertError do -1 else -2 end;
    WriteLn($'''{S}'' -> {N}');
  end;
end.
