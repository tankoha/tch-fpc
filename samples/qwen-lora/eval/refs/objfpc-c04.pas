program Notify;
{$mode objfpc}{$H+}
type
  TNotify = procedure(const Msg: string) of object;

  TSender = class
    OnSend: TNotify;
    procedure Send(const Msg: string);
  end;

  TReceiver = class
    procedure Receive(const Msg: string);
  end;

procedure TSender.Send(const Msg: string);
begin
  if Assigned(OnSend) then
    OnSend(Msg);
end;

procedure TReceiver.Receive(const Msg: string);
begin
  WriteLn('received: ', Msg);
end;

var
  S: TSender;
  R: TReceiver;
begin
  S := TSender.Create;
  R := TReceiver.Create;
  try
    S.OnSend := @R.Receive;
    S.Send('ping');
    S.Send('pong');
  finally
    S.Free;
    R.Free;
  end;
end.
