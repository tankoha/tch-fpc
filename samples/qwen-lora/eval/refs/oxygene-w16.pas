namespace Sentences;

uses
  System.Collections.Generic;

type
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var text := 'Oxygene runs on the dotnet platform';
      var words := text.Split([' ']);
      var rev := new List<String>;
      for i: Integer := words.Length - 1 downto 0 do
        rev.Add(words[i]);
      writeLn(String.Join(' ', rev));
      var caps := new List<String>;
      for each w in words do
        caps.Add(w.Substring(0, 1).ToUpper + w.Substring(1));
      writeLn(String.Join(' ', caps));
    end;
  end;

end.
