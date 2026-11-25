namespace Moves;

type
  InvalidMoveException = public class(Exception);

  Board = public class
  public
    method Move(aFrom, aTo: Integer);
    begin
      if (aFrom < 1) or (aFrom > 9) or (aTo < 1) or (aTo > 9) then
        raise new InvalidMoveException($'square out of board: {aFrom} -> {aTo}');
      if aFrom = aTo then
        raise new InvalidMoveException($'same square {aFrom}');
      writeLn($'moved {aFrom} -> {aTo}');
    end;
  end;

  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var b := new Board;
      for each (f, t) in [(1, 5), (3, 3), (0, 4), (9, 10)] do
        try
          b.Move(f, t);
        except
          on E: InvalidMoveException do
            writeLn('error: ' + E.Message);
        end;
    end;
  end;

end.
