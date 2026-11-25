program Clicks;
{$mode unleashed}
type
  TClick = procedure(const Who: string) of object;

  TForm = class
    procedure OnClick(const Who: string);
  end;

procedure TForm.OnClick(const Who: string);
begin
  WriteLn('clicked by ', Who);
end;

begin
  var F := autofree TForm.Create;
  var H: TClick := @F.OnClick;
  H('Form');
end.
