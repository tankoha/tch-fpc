program Codes;
{$mode unleashed}
begin
  for var Code in [200, 404, 500] do
    WriteLn(match Code of
      200: 'ok';
      404: 'not found';
      _: 'error';
    end);
end.
