program Account;
{$mode objfpc}{$H+}
uses
  SysUtils;
type
  EInsufficientFunds = class(Exception);

  TAccount = class
  private
    FBalance: Integer;
  public
    constructor Create(ABalance: Integer);
    procedure Withdraw(AAmount: Integer);
    property Balance: Integer read FBalance;
  end;

constructor TAccount.Create(ABalance: Integer);
begin
  FBalance := ABalance;
end;

procedure TAccount.Withdraw(AAmount: Integer);
begin
  if AAmount > FBalance then
    raise EInsufficientFunds.CreateFmt('cannot withdraw %d, balance is %d', [AAmount, FBalance]);
  Dec(FBalance, AAmount);
end;

var
  A: TAccount;
begin
  A := TAccount.Create(100);
  try
    A.Withdraw(30);
    try
      A.Withdraw(90);
    except
      on E: EInsufficientFunds do
        WriteLn(E.Message);
    end;
    WriteLn('Balance: ', A.Balance);
  finally
    A.Free;
  end;
end.
