program Calc;
{$mode unleashed}
uses
  SysUtils;

begin
  var Acc := 0;
  for var Line in ['add 5', 'add 3', 'mul 4', 'sub 2', 'noop', 'div 0'] do
  begin
    var P := Pos(' ', Line);
    var Cmd := if P > 0 then Copy(Line, 1, P - 1) else Line;
    var Arg := if P > 0 then StrToInt(Copy(Line, P + 1, MaxInt)) else 0;
    match Cmd of
      'add': Acc += Arg;
      'sub': Acc -= Arg;
      'mul': Acc *= Arg;
      'div':
        if Arg = 0 then
          WriteLn($'skip: {Line}')
        else
          Acc := Acc div Arg;
      _: WriteLn($'unknown: {Line}');
    end;
  end;
  WriteLn($'result: {Acc}');
end.
