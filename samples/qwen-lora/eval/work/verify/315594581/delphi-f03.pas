program MaxGeneric;
{$APPTYPE CONSOLE}
uses
  SysUtils;
type
  TPick = class
  public
    class function MaxOf<T>(const A, B: T; AFirst: Boolean): T; static;
  end;

class function TPick.MaxOf<T>(const A, B: T; AFirst: Boolean): T;
begin
  if AFirst then
    Result := A
  else
    Result := B;
end;

begin
  Writeln('max ', TPick.MaxOf<Integer>(8, 3, True));
end.
