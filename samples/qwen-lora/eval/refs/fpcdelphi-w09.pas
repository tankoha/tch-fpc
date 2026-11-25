program CsvScores;
{$mode delphi}
uses
  SysUtils, Classes;
var
  Rows, Fields: TStringList;
  I, Score, Best, Sum: Integer;
  BestName: string;
  FS: TFormatSettings;
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  Rows := TStringList.Create;
  Fields := TStringList.Create;
  try
    Rows.Delimiter := ';';
    Rows.StrictDelimiter := True;
    Rows.DelimitedText := 'id,name,score;1,Ann,88;2,Bob,72;3,Cid,95';
    Fields.Delimiter := ',';
    Fields.StrictDelimiter := True;
    Best := -1;
    Sum := 0;
    for I := 1 to Rows.Count - 1 do
    begin
      Fields.DelimitedText := Rows[I];
      Score := StrToInt(Fields[2]);
      Inc(Sum, Score);
      if Score > Best then
      begin
        Best := Score;
        BestName := Fields[1];
      end;
    end;
    WriteLn('best: ', BestName, ' ', Best);
    WriteLn('average: ', FormatFloat('0.00', Sum / (Rows.Count - 1), FS));
  finally
    Rows.Free;
    Fields.Free;
  end;
end.
