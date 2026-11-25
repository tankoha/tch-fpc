program Postfix;
{$mode objfpc}{$H+}
uses
  Classes, SysUtils;

function Eval(const AExpr: string): Integer;
var
  Parts: TStringList;
  Stack: array of Integer;
  N, I, A, B: Integer;
  T: string;
begin
  Parts := TStringList.Create;
  try
    Parts.Delimiter := ' ';
    Parts.StrictDelimiter := True;
    Parts.DelimitedText := AExpr;
    Stack := nil;
    SetLength(Stack, Parts.Count);
    N := 0;
    for I := 0 to Parts.Count - 1 do
    begin
      T := Parts[I];
      if (T = '+') or (T = '-') or (T = '*') then
      begin
        B := Stack[N - 1];
        A := Stack[N - 2];
        Dec(N, 2);
        case T[1] of
          '+': Stack[N] := A + B;
          '-': Stack[N] := A - B;
          '*': Stack[N] := A * B;
        end;
      end
      else
        Stack[N] := StrToInt(T);
      Inc(N);
    end;
    Result := Stack[0];
  finally
    Parts.Free;
  end;
end;

procedure Show(const AExpr: string);
begin
  WriteLn(AExpr, ' = ', Eval(AExpr));
end;

begin
  Show('3 4 + 2 *');
  Show('5 1 2 + 4 * + 3 -');
  Show('2 3 4 * + 5 -');
end.
