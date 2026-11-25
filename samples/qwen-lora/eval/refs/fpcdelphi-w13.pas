program Bank;
{$mode delphi}
uses
  SysUtils, Generics.Collections;
type
  TAccount = class
    Balance: Integer;
    constructor Create(ABalance: Integer);
  end;

constructor TAccount.Create(ABalance: Integer);
begin
  Balance := ABalance;
end;

var
  Accounts: TObjectDictionary<string, TAccount>;

procedure Transfer(const AFrom, ATo: string; AAmount: Integer);
begin
  if Accounts[AFrom].Balance < AAmount then
    WriteLn('refused: ', AFrom, '->', ATo, ' ', AAmount)
  else
  begin
    Dec(Accounts[AFrom].Balance, AAmount);
    Inc(Accounts[ATo].Balance, AAmount);
  end;
end;

begin
  Accounts := TObjectDictionary<string, TAccount>.Create([doOwnsValues]);
  try
    Accounts.Add('Alice', TAccount.Create(500));
    Accounts.Add('Bob', TAccount.Create(300));
    Transfer('Alice', 'Bob', 120);
    Transfer('Bob', 'Alice', 50);
    Transfer('Bob', 'Alice', 400);
    WriteLn('Alice ', Accounts['Alice'].Balance);
    WriteLn('Bob ', Accounts['Bob'].Balance);
  finally
    Accounts.Free;
  end;
end.
