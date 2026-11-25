program SimpleStack;
{$APPTYPE CONSOLE}
uses
  System.SysUtils;
type
  TSimpleStack<T> = class
  private
    FItems: TArray<T>;
  public
    procedure Push(const V: T);
    function Pop: T;
    function Count: Integer;
  end;

procedure TSimpleStack<T>.Push(const V: T);
begin
  SetLength(FItems, Length(FItems) + 1);
  FItems[High(FItems)] := V;
end;

function TSimpleStack<T>.Pop: T;
begin
  Result := FItems[High(FItems)];
  SetLength(FItems, Length(FItems) - 1);
end;

function TSimpleStack<T>.Count: Integer;
begin
  Result := Length(FItems);
end;

var
  S: TSimpleStack<string>;
  D: TSimpleStack<Double>;
  FS: TFormatSettings;
begin
  FS := TFormatSettings.Create;
  FS.DecimalSeparator := '.';
  S := TSimpleStack<string>.Create;
  D := TSimpleStack<Double>.Create;
  try
    S.Push('one');
    S.Push('two');
    S.Push('three');
    while S.Count > 0 do
      Writeln(S.Pop);
    D.Push(1.5);
    D.Push(2.5);
    Writeln(FormatFloat('0.0', D.Pop + D.Pop, FS));
  finally
    S.Free;
    D.Free;
  end;
end.
