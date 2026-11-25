namespace Binomial;

uses
  System.Collections.Generic;

type
  Program = class
  private
    class var fCache: Dictionary<String, Int64> := new Dictionary<String, Int64>;
    class method C(n, k: Integer): Int64;
    begin
      if (k = 0) or (k = n) then
        exit 1;
      var key := $'{n},{k}';
      if fCache.ContainsKey(key) then
        exit fCache[key];
      result := C(n - 1, k - 1) + C(n - 1, k);
      fCache[key] := result;
    end;
  public
    class method Main(args: array of String): Int32;
    begin
      for each (n, k) in [(10, 3), (20, 10), (30, 15)] do
        writeLn($'C({n},{k}) = {C(n, k)}');
    end;
  end;

end.
