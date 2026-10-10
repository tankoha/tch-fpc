{ First test of generated ops against the real libtorch (HANDOFF local
  step 5). One op per generated shape: dynamic result (split), out
  parameters (max_dim), results-that-are-inputs (sort_values), tensor list
  argument (cat), Scalar argument (add_scalar), IntListOption none vs empty
  (sum_dim_intlist), DoubleOption + string (pad), Scalar + IntListOption +
  defaults (linalg_norm), LayoutOption (to_sparse), and the native error
  message split (ETorchNativeError.Message / Detail).
  Expected values are worked out by hand in the comments; no Python or
  PyTorch was run to produce them.

  Build (from the repo root, after building build/libtch):
    fpc -Mobjfpc -Sh -Fusrc -FUbuild/generated -Flbuild/libtch \
      -k-rpath=$PWD/build/libtch tests/generated/generated_test.pas \
      -obuild/generated_test
    build/generated_test }
program generated_test;

uses
  SysUtils, Tch;

var
  Failures: Integer = 0;

procedure Check(Cond: Boolean; const What: String);
begin
  if Cond then
    WriteLn('  ok    ', What)
  else
  begin
    WriteLn('  FAIL  ', What);
    Inc(Failures);
  end;
end;

function Near(A, B: Double): Boolean;
begin
  Result := Abs(A - B) < 1e-5;
end;

function ShapeStr(const T: TTensor): String;
var
  S: TInt64Array;
  K: Integer;
begin
  S := T.Shape;
  Result := '[';
  for K := 0 to High(S) do
  begin
    if K > 0 then
      Result := Result + ',';
    Result := Result + IntToStr(S[K]);
  end;
  Result := Result + ']';
end;

{ X = [[0, 1, 2], [3, 4, 5]] as Float }
function MakeX: TTensor;
begin
  Result := TTensor.arange(6, kFloat, TDevice.Cpu).reshape([2, 3]);
end;

procedure TestSplit;
var
  X: TTensor;
  P: TTensorArray;
begin
  WriteLn('split (dynamic result)');
  X := MakeX;
  P := X.split(2, 1);                       // [[0,1],[3,4]] and [[2],[5]]
  Check(Length(P) = 2, 'two pieces');
  Check(ShapeStr(P[0]) = '[2,2]', 'piece 0 shape [2,2]: ' + ShapeStr(P[0]));
  Check(ShapeStr(P[1]) = '[2,1]', 'piece 1 shape [2,1]: ' + ShapeStr(P[1]));
  Check(Near(P[1].DoubleValue([1, 0]), 5), 'piece 1 [1,0] = 5');
end;

procedure TestMaxDim;
var
  X, V, I: TTensor;
begin
  WriteLn('max_dim (out parameters)');
  X := MakeX;
  X.max_dim(1, False, V, I);                // values [2,5], indices [2,2]
  Check(ShapeStr(V) = '[2]', 'values shape [2]');
  Check(Near(V.DoubleValue([0]), 2) and Near(V.DoubleValue([1]), 5),
    'values = [2, 5]');
  Check(I.Kind = kInt64, 'indices are Int64');
  Check((I.Int64Value([0]) = 2) and (I.Int64Value([1]) = 2),
    'indices = [2, 2]');
end;

procedure TestSortValues;
var
  Y, V, I: TTensor;
begin
  WriteLn('sort_values (results are the tensors passed in)');
  Y := TTensor.arange(3, kFloat, TDevice.Cpu);
  Y := (Y * Y).neg;                         // [0, -1, -4]
  V := TTensor.empty([0], kFloat, TDevice.Cpu);
  I := TTensor.empty([0], kInt64, TDevice.Cpu);
  Y.sort_values(V, I);                      // dim -1, ascending: [-4,-1,0], [2,1,0]
  Check(ShapeStr(V) = '[3]', 'values resized to [3]');
  Check(Near(V.DoubleValue([0]), -4) and Near(V.DoubleValue([2]), 0),
    'values = [-4, -1, 0]');
  Check((I.Int64Value([0]) = 2) and (I.Int64Value([2]) = 0),
    'indices = [2, 1, 0]');
end;

procedure TestCat;
var
  X, C: TTensor;
begin
  WriteLn('cat (tensor list, default dim)');
  X := MakeX;
  C := TTensor.cat([X, X + X]);             // dim 0: [4,3], [3,2] = 10
  Check(ShapeStr(C) = '[4,3]', 'shape [4,3]: ' + ShapeStr(C));
  Check(Near(C.DoubleValue([3, 2]), 10), '[3,2] = 10');
  C := TTensor.cat([X, X], 1);
  Check(ShapeStr(C) = '[2,6]', 'dim 1: shape [2,6]');
end;

procedure TestAddScalar;
var
  X, Y: TTensor;
begin
  WriteLn('add_scalar (Scalar from Double and Int64)');
  X := MakeX;
  Y := X.add_scalar(1.5);
  Check(Near(Y.DoubleValue([1, 2]), 6.5), '5 + 1.5 = 6.5');
  Check(Y.Kind = kFloat, 'stays Float');
  Y := TTensor.arange(3, kInt64, TDevice.Cpu).add_scalar(Int64(10));
  Check((Y.Kind = kInt64) and (Y.Int64Value([2]) = 12), 'Int64: 2 + 10 = 12');
end;

procedure TestSumDimIntList;
var
  X, A, B, C: TTensor;
begin
  WriteLn('sum_dim_intlist (IntListOption: none, empty, [1])');
  X := MakeX;
  A := X.sum_dim_intlist(NoIntList);
  B := X.sum_dim_intlist(IntList([]));
  C := X.sum_dim_intlist(IntList([1]), True);
  WriteLn('        none: shape ', ShapeStr(A), ' value ', A.DoubleValue([]):0:3);
  WriteLn('        []  : shape ', ShapeStr(B), ' value ', B.DoubleValue([]):0:3);
  Check((A.Dim = 0) and Near(A.DoubleValue([]), 15), 'none: 0-d, 15');
  // PyTorch: an empty dim list also reduces over all dims for sum.
  Check((B.Dim = 0) and Near(B.DoubleValue([]), 15), 'empty: 0-d, 15');
  Check(ShapeStr(C) = '[2,1]', '[1] keepdim: shape [2,1]');
  Check(Near(C.DoubleValue([0, 0]), 3) and Near(C.DoubleValue([1, 0]), 12),
    '[1] keepdim: [[3], [12]]');
end;

procedure TestPad;
var
  X, P: TTensor;
begin
  WriteLn('pad (string + DoubleOption)');
  X := MakeX;
  P := X.pad([1, 1], 'constant', NoDouble); // last dim 3 -> 5, fill 0
  Check(ShapeStr(P) = '[2,5]', 'shape [2,5]');
  Check(Near(P.DoubleValue([0, 0]), 0) and Near(P.DoubleValue([1, 3]), 5),
    'NoDouble fills 0, data shifted by 1');
  P := X.pad([0, 2], 'constant', 9.0);
  Check(ShapeStr(P) = '[2,5]', 'right pad 2: shape [2,5]');
  Check(Near(P.DoubleValue([1, 4]), 9), 'value 9.0 fills');
end;

procedure TestLinalgNorm;
var
  X, N: TTensor;
begin
  WriteLn('linalg_norm (Scalar + IntListOption + defaults)');
  X := MakeX;
  { dim none on a 2-d input with ord 2 is the matrix 2-norm (largest
    singular value), not the vector norm: X X^T = [[5,14],[14,50]] has
    eigenvalues 54 and 1, so sqrt(54). (First expectation sqrt(55) was
    wrong; libtorch was right.) }
  N := X.linalg_norm(2, NoIntList);
  Check((N.Dim = 0) and Near(N.DoubleValue([]), Sqrt(54)),
    '2-d, dim none: spectral norm sqrt(54) = ' + FloatToStr(N.DoubleValue([])));
  N := X.reshape([6]).linalg_norm(2, NoIntList);
  Check(Near(N.DoubleValue([]), Sqrt(55)), '1-d: vector norm sqrt(55)');
  N := X.linalg_norm(2, IntList([1]), True);  // [[sqrt 5], [sqrt 50]]
  Check(ShapeStr(N) = '[2,1]', 'dim [1] keepdim: shape [2,1]');
  Check(Near(N.DoubleValue([0, 0]), Sqrt(5)) and
    Near(N.DoubleValue([1, 0]), Sqrt(50)), 'dim [1]: [sqrt 5, sqrt 50]');
  N := X.linalg_norm(2, IntList([1]), False, kDouble);
  Check(N.Kind = kDouble, 'dtype kDouble honoured');
end;

procedure TestToSparse;
var
  X, S, D: TTensor;
begin
  WriteLn('to_sparse (LayoutOption)');
  X := MakeX;
  S := X.to_sparse(ltSparse, NoIntList, NoInt64);
  Check(S._nnz = 5, 'ltSparse: nnz = 5 (one zero in X)');
  D := S.to_dense(kNone, False);
  Check(ShapeStr(D) = '[2,3]', 'to_dense: shape [2,3]');
  Check(Near(D.DoubleValue([1, 2]), 5), 'to_dense round trip');
  S := X.to_sparse(ltNone, NoIntList, NoInt64);
  Check(S._nnz = 5, 'ltNone (nullopt): sparse COO, nnz = 5');
end;

procedure TestError;
var
  X: TTensor;
  Raised: Boolean;
begin
  WriteLn('native error: Message vs Detail');
  X := MakeX;
  Raised := False;
  try
    X.matmul(X);                            // [2,3] x [2,3]: shape mismatch
  except
    on E: ETorchNativeError do
    begin
      Raised := True;
      WriteLn('        Message: ', E.Message);
      WriteLn('        Detail : ', Length(E.Detail), ' chars, first line: ',
        Copy(E.Detail, 1, Pos(#10, E.Detail + #10) - 1));
      Check(E.Message <> '', 'Message not empty');
      Check(Pos('Exception raised from', E.Message) = 0,
        'Message has no backtrace');
      Check(Pos(#10'Exception raised from ', E.Detail) > 0,
        'Detail has the backtrace');
      Check(Copy(E.Detail, 1, Length(E.Message)) = E.Message,
        'Detail starts with Message');
    end;
  end;
  Check(Raised, 'ETorchNativeError raised');
end;

begin
  TestSplit;
  TestMaxDim;
  TestSortValues;
  TestCat;
  TestAddScalar;
  TestSumDimIntList;
  TestPad;
  TestLinalgNorm;
  TestToSparse;
  TestError;
  Check(TchLiveHandles = 0, 'no live handles at the end');
  if Failures = 0 then
    WriteLn('OK')
  else
  begin
    WriteLn(Failures, ' FAILED');
    Halt(1);
  end;
end.
