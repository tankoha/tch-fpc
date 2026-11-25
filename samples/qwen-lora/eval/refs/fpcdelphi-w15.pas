program Loggers;
{$mode delphi}
uses
  SysUtils;
type
  ILogger = interface
    ['{2B1E7C0A-4D5F-4C21-8E36-1A9B0C7D3E42}']
    procedure Log(const Msg: string);
  end;

  TPrefixLogger = class(TInterfacedObject, ILogger)
  private
    FPrefix: string;
  public
    constructor Create(const APrefix: string);
    procedure Log(const Msg: string);
  end;

  TCountingLogger = class(TInterfacedObject, ILogger)
  private
    FInner: ILogger;
    FCount: Integer;
  public
    constructor Create(const AInner: ILogger);
    procedure Log(const Msg: string);
  end;

constructor TPrefixLogger.Create(const APrefix: string);
begin
  FPrefix := APrefix;
end;

procedure TPrefixLogger.Log(const Msg: string);
begin
  WriteLn(FPrefix, Msg);
end;

constructor TCountingLogger.Create(const AInner: ILogger);
begin
  FInner := AInner;
end;

procedure TCountingLogger.Log(const Msg: string);
begin
  Inc(FCount);
  FInner.Log(IntToStr(FCount) + ': ' + Msg);
end;

var
  L: ILogger;
begin
  L := TCountingLogger.Create(TPrefixLogger.Create('[app] '));
  L.Log('start');
  L.Log('work');
  L.Log('stop');
end.
