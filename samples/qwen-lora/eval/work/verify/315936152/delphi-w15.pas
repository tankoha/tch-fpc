program Validate;
{$APPTYPE CONSOLE}
uses
  SysUtils;
type
  EValidationError = class(Exception);
  ERangeError2 = class(EValidationError);

procedure ValidateScore(S: Integer);
begin
  if S < 0 then
    raise EValidationError.CreateFmt('%d is negative', [S]);
  if S > 100 then
    raise ERangeError2.CreateFmt('%d is above 100', [S]);
end;

const
  Scores: array[0..2] of Integer = (-5, 50, 150);
var
  Count: Integer;
begin
  Count := 0;
  try
    for var S in Scores do
    begin
      Inc(Count);
      try
        ValidateScore(S);
        Writeln('ok ', S);
      except
        on E: ERangeError2 do
          Writeln('range: ', E.Message);
        on E: EValidationError do
          Writeln('invalid: ', E.Message);
      end;
    end;
  finally
    Writeln('checked ', Count);
  end;
end.
