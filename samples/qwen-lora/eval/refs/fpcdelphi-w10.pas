program Brackets;
{$mode delphi}
uses
  SysUtils, Generics.Collections;

function Balanced(const S: string): Boolean;
var
  St: TStack<Char>;
  C: Char;
begin
  St := TStack<Char>.Create;
  try
    for C in S do
      case C of
        '(', '[', '{': St.Push(C);
        ')', ']', '}':
          begin
            if St.Count = 0 then
              Exit(False);
            if ((C = ')') and (St.Peek <> '(')) or ((C = ']') and (St.Peek <> '[')) or
               ((C = '}') and (St.Peek <> '{')) then
              Exit(False);
            St.Pop;
          end;
      end;
    Result := St.Count = 0;
  finally
    St.Free;
  end;
end;

procedure Check(const S: string);
begin
  if Balanced(S) then
    WriteLn(S, ' balanced')
  else
    WriteLn(S, ' unbalanced');
end;

begin
  Check('(a[b]{c})');
  Check('([)]');
  Check('((x)');
  Check('{[()()]}');
end.
