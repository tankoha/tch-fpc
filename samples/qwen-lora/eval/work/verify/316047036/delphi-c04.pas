program Holder;
{$APPTYPE CONSOLE}
uses
  SysUtils;
type
  THolder<T> = class
    Value: T;
    constructor Create(AValue: T);
  end;

constructor THolder<T>.Create(AValue: T);
begin
  inherited Create;
  Value := AValue;
end;

var
  H: THolder<string>;
begin
  H := THolder<string>.Create('kept');
  Writeln(H.Value);
  H.Free;
end.
