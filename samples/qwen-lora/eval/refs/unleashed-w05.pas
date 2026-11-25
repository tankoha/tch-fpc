program Resources;
{$mode unleashed}

procedure Work;
begin
  WriteLn('open A');
  defer WriteLn('close A');
  WriteLn('open B');
  defer WriteLn('close B');
  WriteLn('open C');
  defer WriteLn('close C');
  WriteLn('working');
end;

begin
  Work;
  WriteLn('done');
end.
