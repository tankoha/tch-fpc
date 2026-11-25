program Ranges;
{$mode delphi}
type
  TRangeEnumerator = class
  private
    FCurrent, FHigh: Integer;
  public
    constructor Create(ALow, AHigh: Integer);
    function MoveNext: Boolean;
    property Current: Integer read FCurrent;
  end;

  TRange = class
  private
    FLow, FHigh: Integer;
  public
    constructor Create(ALow, AHigh: Integer);
    function GetEnumerator: TRangeEnumerator;
  end;

constructor TRangeEnumerator.Create(ALow, AHigh: Integer);
begin
  if Odd(ALow) then
    Inc(ALow);
  FCurrent := ALow - 2;
  FHigh := AHigh;
end;

function TRangeEnumerator.MoveNext: Boolean;
begin
  Inc(FCurrent, 2);
  Result := FCurrent <= FHigh;
end;

constructor TRange.Create(ALow, AHigh: Integer);
begin
  FLow := ALow;
  FHigh := AHigh;
end;

function TRange.GetEnumerator: TRangeEnumerator;
begin
  Result := TRangeEnumerator.Create(FLow, FHigh);
end;

var
  R: TRange;
  I: Integer;
  First: Boolean;
begin
  R := TRange.Create(3, 15);
  try
    First := True;
    for I in R do
    begin
      if not First then
        Write(' ');
      Write(I);
      First := False;
    end;
    WriteLn;
  finally
    R.Free;
  end;
end.
