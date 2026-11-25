program Points;
{$mode objfpc}{$H+}
{$modeswitch advancedrecords}
type
  TPoint = record
    X, Y: Integer;
    class function Make(AX, AY: Integer): TPoint; static;
    function ManhattanTo(const P: TPoint): Integer;
    class operator +(const A, B: TPoint): TPoint;
  end;

class function TPoint.Make(AX, AY: Integer): TPoint;
begin
  Result.X := AX;
  Result.Y := AY;
end;

function TPoint.ManhattanTo(const P: TPoint): Integer;
begin
  Result := Abs(X - P.X) + Abs(Y - P.Y);
end;

class operator TPoint.+(const A, B: TPoint): TPoint;
begin
  Result.X := A.X + B.X;
  Result.Y := A.Y + B.Y;
end;

var
  A, B, C: TPoint;
begin
  A := TPoint.Make(1, 2);
  B := TPoint.Make(4, 6);
  C := A + B;
  WriteLn('(', C.X, ', ', C.Y, ')');
  WriteLn(A.ManhattanTo(B));
end.
