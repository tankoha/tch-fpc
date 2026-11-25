program CountChars;
{$mode delphi}
var
  S: string;
  C: Char;
  V, K, D, Sp: Integer;
begin
  S := 'Pascal 3.2 was released in 2021';
  V := 0;
  K := 0;
  D := 0;
  Sp := 0;
  for C in LowerCase(S) do
    if C in ['a', 'e', 'i', 'o', 'u'] then
      Inc(V)
    else if C in ['a'..'z'] then
      Inc(K)
    else if C in ['0'..'9'] then
      Inc(D)
    else if C = ' ' then
      Inc(Sp);
  WriteLn('vowels ', V);
  WriteLn('consonants ', K);
  WriteLn('digits ', D);
  WriteLn('spaces ', Sp);
end.
