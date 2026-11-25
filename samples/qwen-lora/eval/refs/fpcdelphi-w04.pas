program Perms;
{$mode delphi}
uses
  SysUtils, Classes;

procedure Permute(const Prefix, Rest: string; AOut: TStrings);
var
  I: Integer;
begin
  if Rest = '' then
    AOut.Add(Prefix)
  else
    for I := 1 to Length(Rest) do
      Permute(Prefix + Rest[I], Copy(Rest, 1, I - 1) + Copy(Rest, I + 1, MaxInt), AOut);
end;

var
  L: TStringList;
  S: string;
begin
  L := TStringList.Create;
  try
    Permute('', 'abc', L);
    for S in L do
      WriteLn(S);
    WriteLn(L.Count, ' permutations');
  finally
    L.Free;
  end;
end.
