program Tasks;
{$APPTYPE CONSOLE}
uses
  SysUtils, Generics.Collections;
type
  TTask = class
  public
    Title: string;
    Done: Boolean;
    constructor Create(const ATitle: string);
  end;

constructor TTask.Create(const ATitle: string);
begin
  inherited Create;
  Title := ATitle;
end;

var
  L: TObjectList<TTask>;

procedure MarkDone(const ATitle: string);
begin
  for var T in L do
    if T.Title = ATitle then
      T.Done := True;
end;

function Pending: Integer;
begin
  Result := 0;
  for var T in L do
    if not T.Done then
      Inc(Result);
end;

begin
  L := TObjectList<TTask>.Create;
  try
    for var S in ['plan', 'build', 'test', 'release'] do
      L.Add(TTask.Create(S));
    MarkDone('build');
    MarkDone('test');
    for var T in L do
      if T.Done then
        Writeln('[x] ', T.Title)
      else
        Writeln('[ ] ', T.Title);
    Writeln('pending ', Pending);
  finally
    L.Free;
  end;
end.
