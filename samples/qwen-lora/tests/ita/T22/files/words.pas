namespace Words;

type
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var list := new List<String>;
      list.Add('pear');
      list.Add('apple');
      list.Add('fig');
      list.Sort;
      for each w in list do
        writeLn(w);
      writeLn(list.Count);
    end;
  end;

end.
