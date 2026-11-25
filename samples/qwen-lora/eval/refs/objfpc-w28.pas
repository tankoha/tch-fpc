program GenColl;
{$mode objfpc}{$H+}
uses
  Generics.Collections, SysUtils;
type
  TStrList = specialize TList<string>;
  TLenMap = specialize TDictionary<string, Integer>;
var
  L: TStrList;
  D: TLenMap;
  S: string;
  I: Integer;
begin
  L := TStrList.Create;
  D := TLenMap.Create;
  try
    L.AddRange(['banana', 'apple', 'cherry', 'date', 'fig']);
    L.Sort;
    for I := 0 to L.Count - 1 do
    begin
      if I > 0 then
        Write(',');
      Write(L[I]);
    end;
    WriteLn;
    for S in L do
      D.Add(S, Length(S));
    WriteLn('cherry: ', D['cherry']);
    if D.ContainsKey('grape') then
      WriteLn('grape: yes')
    else
      WriteLn('grape: no');
  finally
    L.Free;
    D.Free;
  end;
end.
