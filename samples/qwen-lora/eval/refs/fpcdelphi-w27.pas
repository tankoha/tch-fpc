program LinkedList;
{$mode delphi}
type
  PNode = ^TNode;
  TNode = record
    Value: Integer;
    Next: PNode;
  end;

procedure Append(var Head: PNode; AValue: Integer);
var
  N, P: PNode;
begin
  New(N);
  N^.Value := AValue;
  N^.Next := nil;
  if Head = nil then
    Head := N
  else
  begin
    P := Head;
    while P^.Next <> nil do
      P := P^.Next;
    P^.Next := N;
  end;
end;

procedure Print(Head: PNode);
begin
  while Head <> nil do
  begin
    Write(Head^.Value);
    if Head^.Next <> nil then
      Write(' -> ');
    Head := Head^.Next;
  end;
  WriteLn;
end;

function Reverse(Head: PNode): PNode;
var
  Next: PNode;
begin
  Result := nil;
  while Head <> nil do
  begin
    Next := Head^.Next;
    Head^.Next := Result;
    Result := Head;
    Head := Next;
  end;
end;

var
  Head, N: PNode;
begin
  Head := nil;
  Append(Head, 3);
  Append(Head, 1);
  Append(Head, 4);
  Append(Head, 1);
  Append(Head, 5);
  Print(Head);
  Head := Reverse(Head);
  Print(Head);
  while Head <> nil do
  begin
    N := Head^.Next;
    Dispose(Head);
    Head := N;
  end;
end.
