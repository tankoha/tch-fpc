namespace Banking;

type
  BankAccount = public class
  public
    property Owner: String; readonly;
    property Balance: Integer read private write;
    constructor(aOwner: String);
    begin
      Owner := aOwner;
    end;
    method Deposit(aAmount: Integer);
    begin
      Balance := Balance + aAmount;
    end;
    method Withdraw(aAmount: Integer);
    begin
      if aAmount > Balance then
        raise new Exception($'cannot withdraw {aAmount}, balance {Balance}');
      Balance := Balance - aAmount;
    end;
  end;

  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var acc := new BankAccount('Ida');
      acc.Deposit(100);
      acc.Deposit(50);
      acc.Withdraw(30);
      try
        acc.Withdraw(500);
      except
        on E: Exception do
          writeLn(E.Message);
      end;
      writeLn($'{acc.Owner}: {acc.Balance}');
    end;
  end;

end.
