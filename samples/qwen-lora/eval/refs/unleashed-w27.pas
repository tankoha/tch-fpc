program EventCounts;
{$mode unleashed}
uses
  SysUtils, Classes, Generics.Collections, Generics.Defaults;
type
  TEntry = TPair<string, Integer>;
  TEntryList = TList<TEntry>;

  TCounterSet = class
  private
    FCounts: TDictionary<string, Integer>;
  public
    constructor Create;
    destructor Destroy; override;
    procedure Hit(const AName: string);
    function Report: TStringList;
  end;

constructor TCounterSet.Create;
begin
  inherited Create;
  FCounts := TDictionary<string, Integer>.Create;
end;

destructor TCounterSet.Destroy;
begin
  FCounts.Free;
  inherited Destroy;
end;

procedure TCounterSet.Hit(const AName: string);
begin
  var N := 0;
  if FCounts.TryGetValue(AName, N) then
    FCounts[AName] := N + 1
  else
    FCounts.Add(AName, 1);
end;

function CompareEntries(const A, B: TEntry): Integer;
begin
  Result := B.Value - A.Value;
  if Result = 0 then
    Result := CompareStr(A.Key, B.Key);
end;

function TCounterSet.Report: TStringList;
begin
  var L := TEntryList.Create(TComparer<TEntry>.Construct(@CompareEntries));
  defer L.Free;
  for var P in FCounts do
    L.Add(P);
  L.Sort;
  Result := TStringList.Create;
  for var P in L do
    Result.Add($'{P.Key} {P.Value}');
end;

begin
  var C := autofree TCounterSet.Create;
  for var E in ['login', 'view', 'view', 'logout', 'view', 'login'] do
    C.Hit(E);
  var R := autofree C.Report;
  for var Line in R do
    WriteLn(Line);
end.
