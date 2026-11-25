program Playlist;
{$APPTYPE CONSOLE}
uses
  SysUtils, Generics.Collections;
type
  TSong = record
    Title: string;
    Seconds: Integer;
  end;

function Song(const ATitle: string; ASeconds: Integer): TSong;
begin
  Result.Title := ATitle;
  Result.Seconds := ASeconds;
end;

function MinSec(S: Integer): string;
begin
  Result := Format('%d:%.2d', [S div 60, S mod 60]);
end;

var
  L: TList<TSong>;
  S: TSong;
  Total: Integer;
begin
  L := TList<TSong>.Create;
  try
    L.Add(Song('Intro', 95));
    L.Add(Song('Main Theme', 241));
    L.Add(Song('Outro', 130));
    Total := 0;
    for S in L do
    begin
      Writeln(S.Title, ' ', MinSec(S.Seconds));
      Inc(Total, S.Seconds);
    end;
    Writeln('total ', MinSec(Total));
  finally
    L.Free;
  end;
end.
