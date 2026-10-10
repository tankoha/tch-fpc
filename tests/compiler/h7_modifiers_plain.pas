{ (H7) Modifiers absolute / alias / index as method names WITHOUT '&'. }
program h7_modifiers_plain;
{$mode objfpc}{$H+}{$modeswitch advancedrecords}
type
  TT = record
    V: Integer;
    function absolute: Integer;
    function alias: Integer;
    function index(const B: TT): Integer;
  end;
function TT.absolute: Integer; begin Result := Abs(V); end;
function TT.alias: Integer; begin Result := V; end;
function TT.index(const B: TT): Integer; begin Result := B.V; end;
var
  T: TT;
begin
  T.V := -3;
  WriteLn(T.absolute, ' ', T.alias, ' ', T.index(T));
end.
