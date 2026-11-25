namespace RoundRobin;

uses
  System.Collections.Generic;

type
  Job = public class
  public
    property Name: String;
    property Left: Integer;
  end;

  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var q := new Queue<Job>;
      q.Enqueue(new Job(Name := 'P1', Left := 5));
      q.Enqueue(new Job(Name := 'P2', Left := 3));
      q.Enqueue(new Job(Name := 'P3', Left := 1));
      var time := 0;
      while q.Count > 0 do begin
        var j := q.Dequeue;
        var run := Math.Min(2, j.Left);
        j.Left := j.Left - run;
        time := time + run;
        writeLn($'{j.Name} runs {run}, {j.Left} left');
        if j.Left = 0 then
          writeLn($'{j.Name} done at {time}')
        else
          q.Enqueue(j);
      end;
    end;
  end;

end.
