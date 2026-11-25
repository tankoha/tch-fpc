namespace Animals;

uses
  System.Collections.Generic;

type
  IAnimal = public interface
    method Name: String;
    method Speak: String;
  end;

  Dog = public class(IAnimal)
  private
    fName: String;
  public
    constructor(aName: String);
    begin
      fName := aName;
    end;
    method Name: String;
    begin
      result := fName;
    end;
    method Speak: String;
    begin
      result := 'woof';
    end;
  end;

  Duck = public class(IAnimal)
  private
    fName: String;
  public
    constructor(aName: String);
    begin
      fName := aName;
    end;
    method Name: String;
    begin
      result := fName;
    end;
    method Speak: String;
    begin
      result := 'quack';
    end;
  end;

  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var animals := new List<IAnimal>;
      animals.Add(new Dog('Rex'));
      animals.Add(new Duck('Donald'));
      var dogs := 0;
      for each a in animals do begin
        writeLn($'{a.Name}: {a.Speak}');
        if a is Dog then
          inc(dogs);
      end;
      writeLn($'dogs {dogs}');
    end;
  end;

end.
