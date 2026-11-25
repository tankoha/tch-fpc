program Config;
{$APPTYPE CONSOLE}
uses
  SysUtils, Generics.Collections, Generics.Defaults;
var
  D: TDictionary<string, string>;
  Keys: TList<string>;
  Part: string;
  KV: TArray<string>;
begin
  D := TDictionary<string, string>.Create;
  Keys := TList<string>.Create(TComparer<string>.Construct(
    function(const A, B: string): Integer
    begin
      Result := CompareStr(A, B);
    end));
  try
    for Part in 'host=example.com;port=8080;debug=true;retries=3'.Split([';']) do
    begin
      KV := Part.Split(['=']);
      D.Add(KV[0], KV[1]);
    end;
    Writeln('port+1 = ', StrToInt(D['port']) + 1);
    Writeln('debug = ', BoolToStr(StrToBool(D['debug']), True));
    for Part in D.Keys do
      Keys.Add(Part);
    Keys.Sort;
    Writeln(string.Join(',', Keys.ToArray));
  finally
    Keys.Free;
    D.Free;
  end;
end.
