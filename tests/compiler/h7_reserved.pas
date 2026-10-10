{ (H7) Op names that are reserved words or modifiers, as record methods. }
program h7_reserved;
{$mode objfpc}{$H+}{$modeswitch advancedrecords}
type
  TT = record
    V: Integer;
    function &div(const B: TT): Integer;
    function &var(dim_: Int64): Integer;
    function &to(device: Integer): Integer;
    function &set(const B: TT): Integer;
    function &repeat(n: Int64): Integer;
    function &absolute: Integer;
    function &alias: Integer;
    function &index(const B: TT): Integer;
    function absolute_plain: Integer;
  end;
function TT.&div(const B: TT): Integer; begin Result := V div B.V; end;
function TT.&var(dim_: Int64): Integer; begin Result := 100 + dim_; end;
function TT.&to(device: Integer): Integer; begin Result := 200 + device; end;
function TT.&set(const B: TT): Integer; begin Result := B.V; end;
function TT.&repeat(n: Int64): Integer; begin Result := V * n; end;
function TT.&absolute: Integer; begin Result := Abs(V); end;
function TT.&alias: Integer; begin Result := V; end;
function TT.&index(const B: TT): Integer; begin Result := 300 + B.V; end;
function TT.absolute_plain: Integer; begin Result := -V; end;
var
  T, U: TT;
begin
  T.V := -12; U.V := 4;
  WriteLn('t.&div(u)    = ', T.&div(U));
  WriteLn('t.&var(1)    = ', T.&var(1));
  WriteLn('t.&to(-1)    = ', T.&to(-1));
  WriteLn('t.&set(u)    = ', T.&set(U));
  WriteLn('t.&repeat(2) = ', T.&repeat(2));
  WriteLn('t.&absolute  = ', T.&absolute);
  WriteLn('t.&alias     = ', T.&alias);
  WriteLn('t.&index(u)  = ', T.&index(U));
end.
