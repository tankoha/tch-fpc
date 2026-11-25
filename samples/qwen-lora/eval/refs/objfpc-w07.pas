program GenericBox;
{$mode objfpc}{$H+}
uses
  SysUtils;
type
  generic TBox<T> = class
  public
    type
      TToText = function(const AValue: T): string;
  private
    FValue: T;
    FToText: TToText;
  public
    constructor Create(const AValue: T; AToText: TToText);
    function Describe: string;
    property Value: T read FValue write FValue;
  end;

constructor TBox.Create(const AValue: T; AToText: TToText);
begin
  FValue := AValue;
  FToText := AToText;
end;

function TBox.Describe: string;
begin
  Result := 'Box(' + FToText(FValue) + ')';
end;

type
  TIntBox = specialize TBox<Integer>;
  TStrBox = specialize TBox<string>;

function IntText(const AValue: Integer): string;
begin
  Result := IntToStr(AValue);
end;

function StrText(const AValue: string): string;
begin
  Result := AValue;
end;

var
  B1: TIntBox;
  B2: TStrBox;
begin
  B1 := TIntBox.Create(42, @IntText);
  B2 := TStrBox.Create('pascal', @StrText);
  try
    WriteLn(B1.Describe);
    WriteLn(B2.Describe);
  finally
    B1.Free;
    B2.Free;
  end;
end.
