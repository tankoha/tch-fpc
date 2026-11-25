program EventBus;
{$mode unleashed}
uses
  SysUtils;
type
  THandler = procedure(const Topic, Msg: string) of object;

  TEventBus = class
  private
    FHandlers: array of THandler;
  public
    procedure Subscribe(H: THandler);
    procedure Publish(const Topic, Msg: string);
  end;

  TListener = class
  private
    FName: string;
  public
    constructor Create(const AName: string);
    procedure Handle(const Topic, Msg: string);
  end;

procedure TEventBus.Subscribe(H: THandler);
begin
  SetLength(FHandlers, Length(FHandlers) + 1);
  FHandlers[High(FHandlers)] := H;
end;

procedure TEventBus.Publish(const Topic, Msg: string);
begin
  for var H in FHandlers do
    H(Topic, Msg);
end;

constructor TListener.Create(const AName: string);
begin
  inherited Create;
  FName := AName;
end;

procedure TListener.Handle(const Topic, Msg: string);
begin
  WriteLn($'{FName} got {Topic}: {Msg}');
end;

begin
  var Bus := autofree TEventBus.Create;
  var A := autofree TListener.Create('audit');
  var M := autofree TListener.Create('mail');
  Bus.Subscribe(@A.Handle);
  Bus.Subscribe(@M.Handle);
  Bus.Publish('order', '#17 created');
  Bus.Publish('order', '#17 paid');
end.
