program Stack;
{$mode delphi}
uses
  Generics.Collections;
type
  TStrStack = TStack<string>;
var
  S: TStrStack;
begin
  S := TStrStack.Create;
  try
    S.Push('first');
    S.Push('second');
    WriteLn(S.Pop);
    WriteLn(S.Pop);
  finally
    S.Free;
  end;
end.
