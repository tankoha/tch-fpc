namespace Greet;

type
  Greeter = public class
  public
    method Greet(aName: String);
    begin
      writeLn('Hi, ' + aName);
    end;
  end;

  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var g := new Greeter;
      g.Greet('Bo');
      g.Greet('Cy');
    end;
  end;

end.
