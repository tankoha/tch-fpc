{ (H3) Scalar operator overloads on a record wrapping an interface. }
program h3_operators;
{$mode objfpc}{$H+}{$modeswitch advancedrecords}
type
  TT = record
    H: IInterface;
    V: Double;
    Via: String;
    class operator +(const A, B: TT): TT;
    class operator +(const A: TT; B: Double): TT;
    class operator +(const A: TT; B: Int64): TT;
    class operator +(A: Double; const B: TT): TT;
    class operator +(A: Int64; const B: TT): TT;
    class operator *(const A: TT; B: Double): TT;
    class operator *(const A: TT; B: Int64): TT;
    class operator *(A: Double; const B: TT): TT;
    class operator *(A: Int64; const B: TT): TT;
    class operator -(const A: TT): TT;
    class operator **(const A: TT; B: Double): TT;
  end;
function Mk(V: Double; const Via: String): TT;
begin
  Result.H := TInterfacedObject.Create; Result.V := V; Result.Via := Via;
end;
class operator TT.+(const A, B: TT): TT; begin Result := Mk(A.V + B.V, 'T+T'); end;
class operator TT.+(const A: TT; B: Double): TT; begin Result := Mk(A.V + B, 'T+Double'); end;
class operator TT.+(const A: TT; B: Int64): TT; begin Result := Mk(A.V + B, 'T+Int64'); end;
class operator TT.+(A: Double; const B: TT): TT; begin Result := Mk(A + B.V, 'Double+T'); end;
class operator TT.+(A: Int64; const B: TT): TT; begin Result := Mk(A + B.V, 'Int64+T'); end;
class operator TT.*(const A: TT; B: Double): TT; begin Result := Mk(A.V * B, 'T*Double'); end;
class operator TT.*(const A: TT; B: Int64): TT; begin Result := Mk(A.V * B, 'T*Int64'); end;
class operator TT.*(A: Double; const B: TT): TT; begin Result := Mk(A * B.V, 'Double*T'); end;
class operator TT.*(A: Int64; const B: TT): TT; begin Result := Mk(A * B.V, 'Int64*T'); end;
class operator TT.-(const A: TT): TT; begin Result := Mk(-A.V, 'neg'); end;
class operator TT.**(const A: TT; B: Double): TT; begin Result := Mk(Exp(B * Ln(A.V)), 'pow'); end;
var
  Failures: Integer = 0;
{ Checks the overload FPC picked (DESIGN.md 4.3, H3 resolved) and the value. }
procedure Show(const What: String; const R: TT; const ExpVia: String; ExpV: Double);
begin
  Write(What, ' = ', R.V:0:2, '  via ', R.Via);
  if (R.Via = ExpVia) and (Abs(R.V - ExpV) < 1e-9) then
    WriteLn('  ok')
  else
  begin
    WriteLn('  FAIL: expected ', ExpVia);
    Inc(Failures);
  end;
end;
var
  T: TT;
  I: Integer = 3;
  S: Single = 0.5;
  B: Byte = 4;
begin
  T := Mk(2, 'mk');
  Show('T + 1      ', T + 1, 'T+Int64', 3);
  Show('T + 1.5    ', T + 1.5, 'T+Double', 3.5);
  Show('1 + T      ', 1 + T, 'Int64+T', 3);
  Show('2 * T      ', 2 * T, 'Int64*T', 4);
  Show('2.5 * T    ', 2.5 * T, 'Double*T', 5);
  Show('T * I (Integer var)', T * I, 'T*Int64', 6);
  Show('T * S (Single var) ', T * S, 'T*Double', 1);
  Show('T * B (Byte var)   ', T * B, 'T*Int64', 8);
  Show('-T         ', -T, 'neg', -2);
  Show('T ** 2     ', T ** 2, 'pow', 4);
  Show('T ** 0.5   ', T ** 0.5, 'pow', Sqrt(2));
  Show('(T + T) * 3', (T + T) * 3, 'T*Int64', 12);
  if Failures > 0 then
    Halt(1);
end.
