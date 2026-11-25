program RingQueue;
{$mode objfpc}{$H+}
uses
  SysUtils;
type
  TRingQueue = class
  private
    FItems: array[0..3] of string;
    FHead, FCount: Integer;
  public
    procedure Enqueue(const S: string);
    function Dequeue: string;
    property Count: Integer read FCount;
  end;

procedure TRingQueue.Enqueue(const S: string);
begin
  if FCount = Length(FItems) then
    raise Exception.Create('queue full, cannot add ' + S);
  FItems[(FHead + FCount) mod Length(FItems)] := S;
  Inc(FCount);
end;

function TRingQueue.Dequeue: string;
begin
  if FCount = 0 then
    raise Exception.Create('queue empty');
  Result := FItems[FHead];
  FHead := (FHead + 1) mod Length(FItems);
  Dec(FCount);
end;

var
  Q: TRingQueue;
begin
  Q := TRingQueue.Create;
  try
    Q.Enqueue('a');
    Q.Enqueue('b');
    Q.Enqueue('c');
    WriteLn(Q.Dequeue);
    WriteLn(Q.Dequeue);
    Q.Enqueue('d');
    Q.Enqueue('e');
    Q.Enqueue('f');
    try
      Q.Enqueue('g');
    except
      on E: Exception do
        WriteLn(E.Message);
    end;
    while Q.Count > 0 do
      WriteLn(Q.Dequeue);
  finally
    Q.Free;
  end;
end.
