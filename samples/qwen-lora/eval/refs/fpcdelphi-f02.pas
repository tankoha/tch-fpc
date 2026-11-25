program Callbacks;
{$mode delphi}
type
  TCallback = procedure(N: Integer) of object;

  TWorker = class
    Count: Integer;
    procedure Step(N: Integer);
  end;

procedure TWorker.Step(N: Integer);
begin
  Count := Count + N;
end;

var
  W: TWorker;
  CB: TCallback;
begin
  W := TWorker.Create;
  CB := W.Step;
  CB(1);
  CB(2);
  WriteLn('done ', W.Count);
  W.Free;
end.
