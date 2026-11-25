program Factory;
{$APPTYPE CONSOLE}
uses
  SysUtils;
type
  TFactory<T: class, constructor> = class
  public
    function Make: T;
  end;

  TApple = class
  public
    constructor Create;
  end;

  TPear = class
  public
    constructor Create;
  end;

function TFactory<T>.Make: T;
begin
  Result := T.Create;
end;

constructor TApple.Create;
begin
  inherited Create;
  Writeln('apple made');
end;

constructor TPear.Create;
begin
  inherited Create;
  Writeln('pear made');
end;

var
  FA: TFactory<TApple>;
  FP: TFactory<TPear>;
  A: TApple;
  P: TPear;
begin
  FA := TFactory<TApple>.Create;
  FP := TFactory<TPear>.Create;
  try
    A := FA.Make;
    P := FP.Make;
    Writeln(A.ClassName);
    Writeln(P.ClassName);
    A.Free;
    P.Free;
  finally
    FA.Free;
    FP.Free;
  end;
end.
