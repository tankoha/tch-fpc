{ (H12) {$modeswitch duplicatelocals} in {$mode objfpc}: a method parameter
  (or out parameter) may share its name with a record member, so generated
  ops can keep PyTorch's parameter names next to hand-written Dim and other
  ops (indices, pow, pow_). Reserved words as parameter names are escaped
  with &. Inside the body the bare name is the parameter, Self.x the member.
  See DESIGN.md 4.4. }
program h12_duplicatelocals;
{$mode objfpc}{$H+}{$modeswitch advancedrecords}
{$modeswitch duplicatelocals}
type
  TT = record
    V: Int64;
    function Dim: Int64;                       // hand-written, declared first
    function indices: Int64;                   // op with the same name as an out param
    function pow(exponent: Int64): Int64;      // op ...
    function pow_(exponent: Int64): Int64;     // ... and its in-place form
    function sum(dim: Int64): Int64;           // parameter shadows method Dim
    procedure max_dim(dim: Int64; out values, indices: Int64);
    function float_power(pow: Int64): Int64;   // parameter shadows methods pow / pow_
    function both(dim: Int64): Int64;          // uses the parameter and Self.Dim
    procedure add_out(&out, &end, &to: Int64); // reserved words, & escaped
    procedure p(result: Int64);                // allowed in a procedure (not in a function)
  end;
function TT.Dim: Int64; begin Result := V; end;
function TT.indices: Int64; begin Result := -1; end;
function TT.pow(exponent: Int64): Int64; begin Result := V * exponent; end;
function TT.pow_(exponent: Int64): Int64; begin Result := V * exponent + 1; end;
function TT.sum(dim: Int64): Int64; begin Result := V + dim; end;
procedure TT.max_dim(dim: Int64; out values, indices: Int64);
begin values := dim * 10; indices := dim * 100; end;
function TT.float_power(pow: Int64): Int64; begin Result := pow + 1000; end;
function TT.both(dim: Int64): Int64; begin Result := dim * 1000 + Self.Dim; end;
procedure TT.add_out(&out, &end, &to: Int64);
begin WriteLn('add_out(&out, &end, &to) = ', &out, ' ', &end, ' ', &to); end;
procedure TT.p(result: Int64); begin WriteLn('p(result)   = ', result); end;
var
  T: TT;
  Vals, Idx: Int64;
begin
  T.V := 7;
  WriteLn('t.sum(2)    = ', T.sum(2));                   // 9
  T.max_dim(3, Vals, Idx);
  WriteLn('t.max_dim(3) = ', Vals, ' ', Idx);            // 30 300
  WriteLn('t.float_power(5) = ', T.float_power(5));      // 1005
  WriteLn('t.both(2)   = ', T.both(2), ' (param*1000 + Self.Dim)'); // 2007
  WriteLn('t.pow(2), t.pow_(2) = ', T.pow(2), ' ', T.pow_(2));      // 14 15
  T.add_out(1, 2, 3);
  T.p(4);
end.
