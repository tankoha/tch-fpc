namespace Stock;

uses
  System.Collections.Generic,
  System.Linq;

type
  Inventory = public class
  private
    fItems: Dictionary<String, Integer> := new Dictionary<String, Integer>;
  public
    method Add(aName: String; aQty: Integer);
    begin
      if fItems.ContainsKey(aName) then
        fItems[aName] := fItems[aName] + aQty
      else
        fItems[aName] := aQty;
    end;
    method Remove(aName: String; aQty: Integer): Boolean;
    begin
      if (not fItems.ContainsKey(aName)) or (fItems[aName] < aQty) then
        exit false;
      fItems[aName] := fItems[aName] - aQty;
      exit true;
    end;
    method Print;
    begin
      for each k in fItems.Keys.OrderBy(k -> k, StringComparer.Ordinal) do
        writeLn($'{k}: {fItems[k]}');
    end;
  end;

  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var inv := new Inventory;
      inv.Add('bolts', 10);
      inv.Add('nuts', 4);
      inv.Add('bolts', 6);
      if not inv.Remove('nuts', 5) then
        writeLn('cannot remove 5 nuts');
      inv.Remove('bolts', 7);
      inv.Print;
    end;
  end;

end.
