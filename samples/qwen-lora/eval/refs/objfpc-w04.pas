program IntStack;
{$mode objfpc}{$H+}
type
  TIntStack = class
  private
    FItems: array of Integer;
    FCount: Integer;
  public
    procedure Push(AValue: Integer);
    function Pop: Integer;
    function IsEmpty: Boolean;
  end;

procedure TIntStack.Push(AValue: Integer);
begin
  if FCount = Length(FItems) then
    SetLength(FItems, FCount * 2 + 4);
  FItems[FCount] := AValue;
  Inc(FCount);
end;

function TIntStack.Pop: Integer;
begin
  Dec(FCount);
  Result := FItems[FCount];
end;

function TIntStack.IsEmpty: Boolean;
begin
  Result := FCount = 0;
end;

var
  S: TIntStack;
  I: Integer;
begin
  S := TIntStack.Create;
  try
    for I := 1 to 5 do
      S.Push(I);
    while not S.IsEmpty do
      WriteLn(S.Pop);
  finally
    S.Free;
  end;
end.
