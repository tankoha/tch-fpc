namespace Matrices;

type
  Matrix = public class
  private
    fData: array of array of Integer;
    fRows, fCols: Integer;
  public
    constructor(aRows, aCols: Integer);
    begin
      fRows := aRows;
      fCols := aCols;
      fData := new array of Integer[aRows];
      for r: Integer := 0 to aRows - 1 do
        fData[r] := new Integer[aCols];
    end;
    method Get(r, c: Integer): Integer;
    begin
      result := fData[r][c];
    end;
    method SetAt(r, c, v: Integer);
    begin
      fData[r][c] := v;
    end;
    method Multiply(other: Matrix): Matrix;
    begin
      result := new Matrix(fRows, other.fCols);
      for r: Integer := 0 to fRows - 1 do
        for c: Integer := 0 to other.fCols - 1 do begin
          var s := 0;
          for k: Integer := 0 to fCols - 1 do
            s := s + fData[r][k] * other.fData[k][c];
          result.SetAt(r, c, s);
        end;
    end;
    property Rows: Integer read fRows;
    property Cols: Integer read fCols;
  end;

  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var a := new Matrix(2, 2);
      var b := new Matrix(2, 2);
      a.SetAt(0, 0, 1); a.SetAt(0, 1, 2); a.SetAt(1, 0, 3); a.SetAt(1, 1, 4);
      b.SetAt(0, 0, 5); b.SetAt(0, 1, 6); b.SetAt(1, 0, 7); b.SetAt(1, 1, 8);
      var m := a.Multiply(b);
      for r: Integer := 0 to m.Rows - 1 do
        writeLn($'{m.Get(r, 0)} {m.Get(r, 1)}');
    end;
  end;

end.
