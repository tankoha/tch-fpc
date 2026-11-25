program Filters;
{$APPTYPE CONSOLE}
uses
  System.SysUtils;
type
  TPredicate<T> = reference to function(const X: T): Boolean;

  TArrayUtils = class
  public
    class function Filter<T>(const A: array of T; const P: TPredicate<T>): TArray<T>; static;
  end;

class function TArrayUtils.Filter<T>(const A: array of T; const P: TPredicate<T>): TArray<T>;
var
  X: T;
  N: Integer;
begin
  SetLength(Result, Length(A));
  N := 0;
  for X in A do
    if P(X) then
    begin
      Result[N] := X;
      Inc(N);
    end;
  SetLength(Result, N);
end;

var
  Evens: TArray<Integer>;
  Long: TArray<string>;
  I: Integer;
begin
  Evens := TArrayUtils.Filter<Integer>([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    function(const X: Integer): Boolean
    begin
      Result := X mod 2 = 0;
    end);
  for I := 0 to High(Evens) do
  begin
    if I > 0 then
      Write(' ');
    Write(Evens[I]);
  end;
  Writeln;
  Long := TArrayUtils.Filter<string>(['tree', 'forest', 'leaf', 'branch'],
    function(const X: string): Boolean
    begin
      Result := X.Length > 4;
    end);
  Writeln(string.Join(' ', Long));
end.
