program WordCount;
{$mode objfpc}{$H+}
uses
  Classes, SysUtils;
var
  Words, Counts: TStringList;
  I, K: Integer;
begin
  Words := TStringList.Create;
  Counts := TStringList.Create;
  try
    Words.Delimiter := ' ';
    Words.StrictDelimiter := True;
    Words.DelimitedText := 'the cat and the dog and the bird';
    Counts.Sorted := True;
    for I := 0 to Words.Count - 1 do
    begin
      K := Counts.IndexOf(Words[I]);
      if K < 0 then
        Counts.AddObject(Words[I], TObject(PtrInt(1)))
      else
        Counts.Objects[K] := TObject(PtrInt(Counts.Objects[K]) + 1);
    end;
    for I := 0 to Counts.Count - 1 do
      WriteLn(Counts[I], ': ', PtrInt(Counts.Objects[I]));
  finally
    Words.Free;
    Counts.Free;
  end;
end.
