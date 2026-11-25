program UndoMemo;
{$APPTYPE CONSOLE}
uses
  SysUtils, Classes;
type
  TMemo = class
  private
    FLines: TStringList;
  public
    constructor Create;
    destructor Destroy; override;
    procedure WriteLine(const S: string);
    procedure Undo;
    function Text: string;
  end;

constructor TMemo.Create;
begin
  inherited Create;
  FLines := TStringList.Create;
end;

destructor TMemo.Destroy;
begin
  FLines.Free;
  inherited;
end;

procedure TMemo.WriteLine(const S: string);
begin
  FLines.Add(S);
end;

procedure TMemo.Undo;
begin
  if FLines.Count > 0 then
    FLines.Delete(FLines.Count - 1);
end;

function TMemo.Text: string;
begin
  Result := '';
  for var I := 0 to FLines.Count - 1 do
  begin
    if I > 0 then
      Result := Result + ' | ';
    Result := Result + FLines[I];
  end;
end;

var
  M: TMemo;
begin
  M := TMemo.Create;
  try
    M.WriteLine('a');
    M.WriteLine('b');
    M.WriteLine('c');
    Writeln('write: ', M.Text);
    M.Undo;
    Writeln('undo: ', M.Text);
    M.WriteLine('d');
    Writeln('write: ', M.Text);
    M.Undo;
    M.Undo;
    Writeln('undo2: ', M.Text);
  finally
    M.Free;
  end;
end.
