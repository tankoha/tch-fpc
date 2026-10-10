{ Ownership tests for TTensor (HANDOFF step 6, H1/H6/H8).

  Build (from the repo root, after building build/libtch):
    fpc -Mobjfpc -Sh -Fusrc -FUbuild/ownership -Flbuild/libtch \
      -k-rpath=$PWD/build/libtch tests/ownership/ownership_test.pas \
      -obuild/ownership_test
    build/ownership_test }
program ownership_test;

uses
  SysUtils, Tch;

const
  CpuDevice = -1;

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

function RssKiB: Int64;
var
  F: TextFile;
  Size, Resident: Int64;
begin
  AssignFile(F, '/proc/self/statm');
  Reset(F);
  Read(F, Size, Resident);
  CloseFile(F);
  Result := Resident * 4;
end;

function Ones23: TTensor;
begin
  Result := TTensor.ones([2, 3], kFloat, CpuDevice);
end;

procedure TestAliasing;
var
  A, B: TTensor;
  Base: Integer;
begin
  WriteLn('Aliasing: B := A, then A := Default(TTensor)');
  Base := TchLiveHandles;
  A := Ones23;
  B := A;
  Check(TchLiveHandles = Base + 1, 'B := A shares one handle');
  Check(B.SameHandle(A), 'SameHandle');
  A := Default(TTensor);
  Check(A.IsNil, 'A is nil');
  Check((TchLiveHandles = Base + 1) and (B.DoubleValue([1, 2]) = 1.0),
    'B still alive and readable');
  B := Default(TTensor);
  Check(TchLiveHandles = Base, 'last reference released');
end;

procedure TestSelfAssign;
var
  A, B: TTensor;
begin
  WriteLn('Self-assignment: A := A + B');
  A := Ones23;
  B := Ones23;
  A := A + B;
  Check(A.DoubleValue([0, 0]) = 2.0, 'A = 2 after A := A + B');
  A := A * A;
  Check(A.DoubleValue([1, 2]) = 4.0, 'A = 4 after A := A * A');
end;

{ (H1) When are the temporaries of (A + B) * C released? }
procedure H1Statement(Base: Integer);
var
  A, B, C, D, E: TTensor;
  I: Integer;
begin
  A := Ones23; B := Ones23; C := Ones23;
  WriteLn('  inside routine, after creating A, B, C: live = ',
    TchLiveHandles - Base);
  D := (A + B) * C;
  WriteLn('  after D := (A + B) * C:               live = ',
    TchLiveHandles - Base, '  (4 = temp freed, 5 = temp kept)');
  E := (A + B) * C;
  WriteLn('  after E := (A + B) * C:               live = ',
    TchLiveHandles - Base);
  for I := 1 to 3 do
  begin
    D := (A + B) * C;
    WriteLn('  loop iteration ', I, ', after D := ...:      live = ',
      TchLiveHandles - Base);
  end;
  WriteLn('  (A + B) * C as a call argument, value = ',
    FloatToStr(((A + B) * C).DoubleValue([0, 0])), ', live after = ',
    TchLiveHandles - Base);
end;

procedure TestH1;
var
  Base: Integer;
begin
  WriteLn('(H1) lifetime of expression temporaries');
  Base := TchLiveHandles;
  H1Statement(Base);
  Check(TchLiveHandles = Base, 'everything released at routine exit');
end;

{ Temporaries live until the routine exits (H1), so handle counts are
  checked after the routine returns, or as a delta across a statement that
  creates no temporaries. }
procedure LoopRss(N: Integer);
var
  A, B, C, D: TTensor;
  I, Live1, Live2: Integer;
  Rss0, Rss1, Rss2: Int64;
begin
  A := Ones23; B := Ones23; C := Ones23;
  for I := 1 to N div 10 do         // warm-up: allocator pools settle
    D := (A + B) * C;
  Rss0 := RssKiB;
  for I := 1 to N div 2 do
    D := (A + B) * C;
  Rss1 := RssKiB;
  Live1 := TchLiveHandles;
  for I := 1 to N div 2 do
    D := (A + B) * C;
  Rss2 := RssKiB;
  Live2 := TchLiveHandles;
  WriteLn(Format('  RSS KiB: after warm-up %d, +%d iter %d, +%d iter %d',
    [Rss0, N div 2, Rss1, N div 2, Rss2]));
  Check(Live1 = Live2, Format('live handles constant (%d)', [Live2]));
  Check(Rss2 - Rss1 < 1024, 'RSS flat over the second half (< 1 MiB)');
  Check(D.DoubleValue([0, 0]) = 2.0, 'D = 2');
end;

procedure TestLoopRss;
const
  N = 1000000;
var
  Base: Integer;
begin
  WriteLn('Loop: ', N, ' x D := (A + B) * C');
  Base := TchLiveHandles;
  LoopRss(N);
  Check(TchLiveHandles = Base, 'all released at routine exit');
end;

{ (H6) open-array constructor of TTensor passed to cat. }
procedure H6Body(Base: Integer);
var
  A, B, R: TTensor;
  I: Integer;
  Rss0, Rss1: Int64;
  S: TInt64Array;
begin
  A := Ones23; B := Ones23;
  R := TTensor.cat([A, B], 0);
  S := R.Shape;
  Check((Length(S) = 2) and (S[0] = 4) and (S[1] = 3), 'shape [4, 3]');
  WriteLn('  live after cat([A, B]) = ', TchLiveHandles - Base);
  R := TTensor.cat([A + B, B, A * B], 1);
  S := R.Shape;
  Check((S[0] = 2) and (S[1] = 9), 'cat of temporaries: shape [2, 9]');
  WriteLn('  live after cat([A + B, B, A * B]) = ', TchLiveHandles - Base);
  Rss0 := RssKiB;
  for I := 1 to 200000 do
    R := TTensor.cat([A, B, A + B], 0);
  Rss1 := RssKiB;
  WriteLn(Format('  RSS KiB over 200000 cats: %d -> %d', [Rss0, Rss1]));
  Check(Rss1 - Rss0 < 1024, 'RSS flat (< 1 MiB)');
end;

procedure TestH6;
var
  Base: Integer;
begin
  WriteLn('(H6) TTensor.cat([A, B], 0)');
  Base := TchLiveHandles;
  H6Body(Base);
  Check(TchLiveHandles = Base, 'all released at routine exit (no leak, '
    + 'no double free)');
end;

{ (H8) out parameters. }
procedure H8Body;
var
  T, V, I, Old: TTensor;
  L0: Integer;
  Raised: Boolean;
begin
  T := Ones23;
  Raised := False;
  try
    T.max_dim(1, False, T, I);
  except
    on E: ETorchError do
    begin
      Raised := True;
      WriteLn('  T.max_dim(1, False, T, I) raised ', E.ClassName, ': ',
        E.Message);
    end;
  end;
  Check(Raised, 'receiver passed as out raises ETorchError, no crash');
  Check(T.IsNil, 'T was finalised on entry');

  T := Ones23;
  T.max_dim(1, False, V, I);           // V, I now hold values
  Old := V;                            // second reference to the old V
  L0 := TchLiveHandles;
  T.max_dim(1, False, V, I);
  Check(not V.SameHandle(Old), 'V now holds a new handle');
  Check(TchLiveHandles - L0 = 1, 'delta +1: new V and I, old I freed, '
    + 'old V kept by Old');
  Old := Default(TTensor);
  Check(TchLiveHandles - L0 = 0, 'old V freed once Old drops it');
  L0 := TchLiveHandles;
  T.max_dim(1, False, V, I);
  Check(TchLiveHandles - L0 = 0,
    'delta 0: previous V and I released by the out call');
  Check(V.Dim = 1, 'values has 1 dim');
end;

procedure TestH8;
var
  Base: Integer;
begin
  WriteLn('(H8) out parameter semantics');
  Base := TchLiveHandles;
  H8Body;
  Check(TchLiveHandles = Base, 'all released at routine exit');
end;

begin
  TestAliasing;
  TestSelfAssign;
  TestH1;
  TestH6;
  TestH8;
  TestLoopRss;
  Check(TchLiveHandles = 0, 'no live handles at the end');
  if Failures = 0 then
    WriteLn('OK')
  else
  begin
    WriteLn('FAILED (', Failures, ')');
    Halt(1);
  end;
end.
