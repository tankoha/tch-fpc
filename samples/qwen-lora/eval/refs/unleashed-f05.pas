program Words;
{$mode unleashed}

function Name(N: Integer): string;
begin
  Result := if N = 10 then 'ten' else 'other';
end;

begin
  WriteLn(Name(10));
  WriteLn(Name(3));
end.
