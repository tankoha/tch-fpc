program Grades;
{$mode unleashed}
uses
  SysUtils;

function Letter(S: Integer): string;
begin
  Result := match
    S >= 90: 'A';
    S >= 80: 'B';
    S >= 70: 'C';
    S >= 60: 'D';
    _: 'F';
  end;
end;

begin
  var Sum := 0;
  var N := 0;
  for var Item in 'Ann:91,Ben:58,Cara:77,Dev:84,Eli:63'.Split([',']) do
  begin
    var Parts := Item.Split([':']);
    var Score := StrToInt(Parts[1]);
    WriteLn($'{Parts[0]}: {Score} {Letter(Score)}');
    Sum += Score;
    Inc(N);
  end;
  var FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  WriteLn('average ', FormatFloat('0.0', Sum / N, FS));
end.
