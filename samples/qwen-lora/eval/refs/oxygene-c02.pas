namespace People;

type
  Person = public class
  private
    fName: String;
    fAge: Integer;
  public
    constructor(aName: String; aAge: Integer);
    begin
      fName := aName;
      fAge := aAge;
    end;
    method Describe: String;
    begin
      result := $'{fName} is {fAge}';
    end;
  end;

  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var p := new Person('Lin', 34);
      writeLn(p.Describe);
    end;
  end;

end.
