program Vending;
{$mode unleashed}
uses
  SysUtils, TypInfo;
type
  TState = (Idle, HasCoin, Dispensing);

begin
  var State := Idle;
  for var Ev in ['coin', 'push', 'coin', 'coin', 'push', 'push'] do
  begin
    match State of
      Idle:
        if Ev = 'coin' then
          State := HasCoin
        else
          WriteLn('insert coin first');
      HasCoin:
        if Ev = 'coin' then
          WriteLn('refund extra coin')
        else
        begin
          State := Dispensing;
          WriteLn('dispense');
          State := Idle;
        end;
      _: State := Idle;
    end;
    WriteLn($'{Ev} -> {GetEnumName(TypeInfo(TState), Ord(State))}');
  end;
end.
