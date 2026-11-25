program Payroll;
{$APPTYPE CONSOLE}
uses
  SysUtils, Generics.Collections;
type
  TEmployee = class
  private
    FName: string;
  public
    constructor Create(const AName: string);
    function MonthlyPay: Double; virtual; abstract;
    property Name: string read FName;
  end;

  TSalaried = class(TEmployee)
  private
    FAnnual: Double;
  public
    constructor Create(const AName: string; AAnnual: Double);
    function MonthlyPay: Double; override;
  end;

  THourly = class(TEmployee)
  private
    FHours, FRate: Double;
  public
    constructor Create(const AName: string; AHours, ARate: Double);
    function MonthlyPay: Double; override;
  end;

constructor TEmployee.Create(const AName: string);
begin
  inherited Create;
  FName := AName;
end;

constructor TSalaried.Create(const AName: string; AAnnual: Double);
begin
  inherited Create(AName);
  FAnnual := AAnnual;
end;

function TSalaried.MonthlyPay: Double;
begin
  Result := FAnnual / 12;
end;

constructor THourly.Create(const AName: string; AHours, ARate: Double);
begin
  inherited Create(AName);
  FHours := AHours;
  FRate := ARate;
end;

function THourly.MonthlyPay: Double;
begin
  Result := FHours * FRate;
end;

var
  Staff: TObjectList<TEmployee>;
  E: TEmployee;
  Total: Double;
  FS: TFormatSettings;
begin
  FS := TFormatSettings.Create;
  FS.DecimalSeparator := '.';
  Staff := TObjectList<TEmployee>.Create;
  try
    Staff.Add(TSalaried.Create('Ann', 60000));
    Staff.Add(THourly.Create('Bo', 160, 25));
    Staff.Add(TSalaried.Create('Cy', 48000));
    Total := 0;
    for E in Staff do
    begin
      Writeln(E.Name, ': ', FormatFloat('0.00', E.MonthlyPay, FS));
      Total := Total + E.MonthlyPay;
    end;
    Writeln('total: ', FormatFloat('0.00', Total, FS));
  finally
    Staff.Free;
  end;
end.
