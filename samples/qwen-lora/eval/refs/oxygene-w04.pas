namespace Shapes;

uses
  System.Collections.Generic,
  System.Globalization;

type
  Shape = public abstract class
  public
    method Area: Double; abstract;
    method Name: String; abstract;
    method Describe: String;
    begin
      result := Name + ' ' + Area.ToString('0.00', CultureInfo.InvariantCulture);
    end;
  end;

  Rectangle = public class(Shape)
  private
    fW, fH: Double;
  public
    constructor(aW, aH: Double);
    begin
      fW := aW;
      fH := aH;
    end;
    method Area: Double; override;
    begin
      result := fW * fH;
    end;
    method Name: String; override;
    begin
      result := 'Rectangle';
    end;
  end;

  Circle = public class(Shape)
  private
    fR: Double;
  public
    constructor(aR: Double);
    begin
      fR := aR;
    end;
    method Area: Double; override;
    begin
      result := Math.PI * fR * fR;
    end;
    method Name: String; override;
    begin
      result := 'Circle';
    end;
  end;

  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var shapes := new List<Shape>;
      shapes.Add(new Rectangle(3, 4));
      shapes.Add(new Circle(1.5));
      var total := 0.0;
      for each s in shapes do begin
        writeLn(s.Describe);
        total := total + s.Area;
      end;
      writeLn('total ' + total.ToString('0.00', CultureInfo.InvariantCulture));
    end;
  end;

end.
