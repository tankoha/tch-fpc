program Wrap;
{$mode unleashed}
uses
  SysUtils;

begin
  var Text := 'Unleashed Pascal adds inline variables, match expressions, tuples and async to the objfpc dialect.';
  var Line := '';
  for var W in Text.Split([' ']) do
  begin
    if Line = '' then
      Line := W
    else if Length(Line) + 1 + Length(W) <= 24 then
      Line += ' ' + W
    else
    begin
      WriteLn(Line);
      Line := W;
    end;
  end;
  if Line <> '' then
    WriteLn(Line);
end.
