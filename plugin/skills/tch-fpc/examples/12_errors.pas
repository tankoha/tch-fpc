{ Errors: libtorch's failures arrive as ETorchNativeError (Message is the
  first line, Detail the full text with the C++ backtrace); misuse caught
  on the Pascal side is ETorchError. }
program errors;

{$mode objfpc}{$H+}

uses
  SysUtils, Tch;

var
  A, B, N: TTensor;
begin
  A := TTensor.ones([2, 3]);
  B := TTensor.ones([4, 5]);
  try
    A.matmul(B);
  except
    on E: ETorchNativeError do
      WriteLn('native: ', Pos('cannot be multiplied', E.Message) > 0,
        ', detail longer: ', Length(E.Detail) >= Length(E.Message));
  end;
  try
    A.add(N);                    // N holds no tensor
  except
    on E: ETorchNativeError do
      WriteLn('unexpected native error');
    on E: ETorchError do
      WriteLn('pascal side: ', E.ClassName);
  end;
  WriteLn('still running');
end.
