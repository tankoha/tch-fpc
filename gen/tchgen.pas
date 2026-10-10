{ tchgen: Pascal emitter for tch-fpc (DESIGN.md section 5).

  Reads gen/ops.tsv (written by gen/gen.ml) and src/Tch.pas (for the
  hand-written TTensor member names) and writes
    src/TchSysGenerated.pas            L2: external atg_* declarations
    src/TchTensorGenerated_intf.inc    L3: method declarations inside TTensor
    src/TchTensorGenerated_impl.inc    L3: method bodies

  Usage (from the repo root):
    instantfpc gen/tchgen.pas            or    fpc gen/tchgen.pas && gen/tchgen
    tchgen [ops.tsv] [src dir]           defaults: gen/ops.tsv, src

  All Pascal-specific policy lives here (DESIGN.md 4.3, 4.4): parameter
  types per gen.ml argument kind, the & escape of reserved words, result_
  for "result", trailing default-parameter runs, out parameters for
  multi-tensor results (or none when the results are the tensors passed
  in), nil-handle checks, temporaries for Scalar and tensor-list
  arguments, and the collision checks. Anything unexpected in ops.tsv is
  an error (exit code 1), never silently skipped. }
program tchgen;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes;

type
  TArgKind = (akBool, akInt64, akInt64Option, akDouble, akDoubleOption,
    akTensor, akTensorOption, akIntList, akIntListOption, akDoubleList,
    akTensorOptList, akTensorList, akTensorOptions, akScalar, akScalarType,
    akScalarTypeOption, akDevice, akString, akLayout, akLayoutOption);

  TArg = record
    Name: String;        // verbatim YAML name
    Kind: TArgKind;
    HasDefault: Boolean;
    Default: String;     // canonical text from gen.ml (see default_text)
  end;

  TResultShape = (rsNothing, rsFixed, rsDynamic, rsBool, rsInt64, rsDouble);

  TOp = record
    Name: String;              // exported name; C function is atg_<Name>
    Receiver: String;          // 'self', 'input' or '-'
    Shape: TResultShape;
    NFixed: Integer;           // for rsFixed
    Args: array of TArg;       // C order, receiver included
    Returns: array of String;  // rsFixed with NFixed >= 2
    OutLess: Boolean;          // every result is also an argument: no out params
    DefaultFrom: Integer;      // first index of the trailing default run; Length(Args) if none
  end;

const
  ArgKindNames: array[TArgKind] of String = ('Bool', 'Int64', 'Int64Option',
    'Double', 'DoubleOption', 'Tensor', 'TensorOption', 'IntList',
    'IntListOption', 'DoubleList', 'TensorOptList', 'TensorList',
    'TensorOptions', 'Scalar', 'ScalarType', 'ScalarTypeOption', 'Device',
    'String', 'Layout', 'LayoutOption');

  { Argument kinds that may carry a Pascal default (ordinal, real, string,
    enum). Records and open arrays cannot (measured, FPC 3.2.2). }
  DefaultableKinds = [akBool, akInt64, akDouble, akString, akScalarType,
    akScalarTypeOption, akLayout, akLayoutOption];

  { FPC reserved words and modifiers (Turbo Pascal, Object Pascal, FPC),
    lower case. Any op or parameter name in this list is written with the
    & escape. Over-escaping a modifier is harmless (H7). }
  ReservedWords: array[0..150] of String = (
    'absolute', 'abstract', 'alias', 'and', 'array', 'as', 'asm', 'assembler',
    'begin', 'bitpacked', 'break', 'case', 'cdecl', 'class', 'const',
    'constref', 'constructor', 'continue', 'cppdecl', 'cvar', 'default',
    'deprecated', 'destructor', 'dispose', 'div', 'do', 'downto', 'dynamic',
    'else', 'end', 'enumerator', 'except', 'exit', 'experimental', 'export',
    'exports', 'external', 'false', 'far', 'far16', 'file', 'finalization',
    'finally', 'for', 'forward', 'function', 'generic', 'goto', 'helper',
    'if', 'implementation', 'implements', 'in', 'index', 'inherited',
    'initialization', 'inline', 'interface', 'interrupt', 'iocheck', 'is',
    'label', 'library', 'local', 'message', 'mod', 'name', 'near', 'new',
    'nil', 'nodefault', 'noreturn', 'nostackframe', 'not', 'object', 'of',
    'oldfpccall', 'on', 'operator', 'optional', 'or', 'otherwise', 'out',
    'overload', 'override', 'packed', 'pascal', 'platform', 'private',
    'procedure', 'program', 'property', 'protected', 'public', 'published',
    'raise', 'read', 'record', 'register', 'reintroduce', 'repeat',
    'resourcestring', 'result', 'safecall', 'saveregisters', 'set',
    'shl', 'shr', 'softfloat', 'specialize', 'static', 'stdcall', 'stored',
  { 'self' is left out: it names the receiver in L2 declarations }
    'strict', 'string', 'then', 'threadvar', 'to', 'true', 'try', 'type',
    'unaligned', 'unimplemented', 'unit', 'until', 'uses', 'var', 'varargs',
    'virtual', 'while', 'winapi', 'with', 'write', 'xor',
    'vectorcall', 'ms_abi_default', 'ms_abi_cdecl', 'sysv_abi_default',
    'sysv_abi_cdecl', 'sealed', 'final', 'hardfloat', 'mwpascal', 'syscall',
    'section', 'noinline', 'objcclass', 'objcprotocol', 'objccategory',
    'weakexternal', 'nested');

var
  Ops: array of TOp;
  HandWritten: TStringList;   // TTensor members in Tch.pas, lower case
  { hand-written members declared with `overload;` only: deliberate
    overloads (of generated ops or of each other), not collisions }
  Overloads: TStringList;
  OpNames: TStringList;       // generated op names, lower case
  OpsTsv, SrcDir: String;

procedure Fail(const Msg: String);
begin
  WriteLn(StdErr, 'tchgen: ', Msg);
  Halt(1);
end;

function IsReserved(const S: String): Boolean;
var
  L: String;
  I: Integer;
begin
  L := LowerCase(S);
  for I := Low(ReservedWords) to High(ReservedWords) do
    if ReservedWords[I] = L then
      Exit(True);
  Result := False;
end;

{ Pascal spelling of a verbatim op or parameter name. }
function Ident(const S: String): String;
begin
  if LowerCase(S) = 'result' then
    Result := 'result_'
  else if IsReserved(S) then
    Result := '&' + S
  else
    Result := S;
end;

function Split(const S: String; Sep: Char): TStringArray;
var
  Start, I, N: Integer;
begin
  Result := nil;
  N := 0;
  Start := 1;
  for I := 1 to Length(S) + 1 do
    if (I > Length(S)) or (S[I] = Sep) then
    begin
      SetLength(Result, N + 1);
      Result[N] := Copy(S, Start, I - Start);
      Inc(N);
      Start := I + 1;
    end;
end;

function ParseKind(const S: String): TArgKind;
var
  K: TArgKind;
begin
  for K := Low(TArgKind) to High(TArgKind) do
    if ArgKindNames[K] = S then
      Exit(K);
  Fail('unknown argument kind "' + S + '"');
  Result := akBool;
end;

procedure ReadOps(const FileName: String);
var
  Lines: TStringList;
  F: TStringArray;
  I, J, NArgs, P: Integer;
  Op: TOp;
  A: TArg;
  Field, Rest: String;
begin
  Lines := TStringList.Create;
  try
    Lines.LoadFromFile(FileName);
    Ops := nil;
    for I := 0 to Lines.Count - 1 do
    begin
      if (Lines[I] = '') or (Lines[I][1] = '#') then
        Continue;
      F := Split(Lines[I], #9);
      if Length(F) < 3 then
        Fail('line ' + IntToStr(I + 1) + ': fewer than 3 columns');
      Op := Default(TOp);
      Op.Name := F[0];
      Op.Receiver := F[1];
      if (Op.Receiver <> 'self') and (Op.Receiver <> 'input') and (Op.Receiver <> '-') then
        Fail(Op.Name + ': bad receiver ' + Op.Receiver);
      if F[2] = 'nothing' then Op.Shape := rsNothing
      else if F[2] = 'dynamic' then Op.Shape := rsDynamic
      else if F[2] = 'bool' then Op.Shape := rsBool
      else if F[2] = 'int64' then Op.Shape := rsInt64
      else if F[2] = 'double' then Op.Shape := rsDouble
      else if Copy(F[2], 1, 5) = 'fixed' then
      begin
        Op.Shape := rsFixed;
        Op.NFixed := StrToInt(Copy(F[2], 6, MaxInt));
        if (Op.NFixed < 1) or (Op.NFixed > 9) then
          Fail(Op.Name + ': bad result ' + F[2]);
      end
      else
        Fail(Op.Name + ': bad result ' + F[2]);
      NArgs := 0;
      for J := 3 to High(F) do
      begin
        Field := F[J];
        if Copy(Field, 1, 2) = '->' then
        begin
          Op.Returns := Split(Copy(Field, 3, MaxInt), ',');
          if J <> High(F) then
            Fail(Op.Name + ': -> must be the last column');
          Break;
        end;
        P := Pos(':', Field);
        if P < 2 then
          Fail(Op.Name + ': bad argument field "' + Field + '"');
        A := Default(TArg);
        A.Name := Copy(Field, 1, P - 1);
        Rest := Copy(Field, P + 1, MaxInt);
        P := Pos('=', Rest);
        if P > 0 then
        begin
          A.HasDefault := True;
          A.Default := Copy(Rest, P + 1, MaxInt);
          Rest := Copy(Rest, 1, P - 1);
        end;
        A.Kind := ParseKind(Rest);
        if Copy(A.Name, Length(A.Name) - 1, 2) = '__' then
          Fail(Op.Name + ': argument name ' + A.Name + ' ends in __ (reserved for generated locals)');
        SetLength(Op.Args, NArgs + 1);
        Op.Args[NArgs] := A;
        Inc(NArgs);
      end;
      if Op.Shape = rsFixed then
      begin
        if (Op.NFixed >= 2) and (Length(Op.Returns) <> Op.NFixed) then
          Fail(Op.Name + ': fixed' + IntToStr(Op.NFixed) + ' needs ' + IntToStr(Op.NFixed) + ' return names');
        if (Op.NFixed = 1) and (Length(Op.Returns) <> 0) then
          Fail(Op.Name + ': fixed1 must not list return names');
      end
      else if Length(Op.Returns) <> 0 then
        Fail(Op.Name + ': return names on a non-fixed result');
      if (Op.Receiver <> '-') then
      begin
        P := -1;
        for J := 0 to High(Op.Args) do
          if (Op.Args[J].Name = Op.Receiver) and (Op.Args[J].Kind = akTensor) then
            P := J;
        if P < 0 then
          Fail(Op.Name + ': receiver ' + Op.Receiver + ' is not a Tensor argument');
      end;
      SetLength(Ops, Length(Ops) + 1);
      Ops[High(Ops)] := Op;
    end;
  finally
    Lines.Free;
  end;
end;

{ Results that are also arguments: the op fills tensors passed in (the
  "out=" variants). All or nothing; a partial overlap is unexpected. }
procedure ClassifyOutLess(var Op: TOp);
var
  I, J, Hits: Integer;
begin
  Op.OutLess := False;
  if (Op.Shape <> rsFixed) or (Op.NFixed < 2) then
    Exit;
  Hits := 0;
  for I := 0 to High(Op.Returns) do
    for J := 0 to High(Op.Args) do
      if SameText(Op.Returns[I], Op.Args[J].Name) then
      begin
        Inc(Hits);
        Break;
      end;
  if Hits = Length(Op.Returns) then
    Op.OutLess := True
  else if Hits > 0 then
    Fail(Op.Name + ': some but not all result names are argument names');
end;

{ Trailing run of defaultable arguments that have a default. Ops with out
  parameters get none (defaults must be the last parameters). }
procedure ComputeDefaultRun(var Op: TOp);
var
  I: Integer;
begin
  Op.DefaultFrom := Length(Op.Args);
  if (Op.Shape = rsFixed) and (Op.NFixed >= 2) and not Op.OutLess then
    Exit;
  { "None" on a kind whose Pascal type cannot express it (bool?, str? ...
    reach C as a plain bool / empty string) gets no default: the caller
    chooses a value explicitly, as in tch-rs. }
  for I := High(Op.Args) downto 0 do
    if Op.Args[I].HasDefault and (Op.Args[I].Kind in DefaultableKinds)
       and (Op.Args[I].Name <> Op.Receiver)
       and ((Op.Args[I].Default <> '::std::nullopt')
            or (Op.Args[I].Kind in [akScalarTypeOption, akLayoutOption])) then
      Op.DefaultFrom := I
    else
      Break;
end;

{ Pascal literal for a default. Unknown texts are errors so that a new
  libtorch surfaces them here, not at compile time. }
function PasDefault(const Op: TOp; const A: TArg): String;
var
  S, R: String;
  I: Integer;
  D: Double;
  FS: TFormatSettings;
begin
  Result := '';
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  case A.Kind of
    akBool:
      if A.Default = 'true' then Result := 'True'
      else if A.Default = 'false' then Result := 'False';
    akInt64:
      if A.Default = 'at::Reduction::Mean' then Result := '1'
      else if A.Default = 'at::Reduction::None' then Result := '0'
      else if A.Default = 'at::Reduction::Sum' then Result := '2'
      else if (A.Default <> '') and (StrToInt64Def(A.Default, Low(Int64)) <> Low(Int64)) then
        Result := A.Default;
    akDouble:
      if (A.Default <> '') and TryStrToFloat(A.Default, D, FS) then
        Result := A.Default;   // canonical %.16g text is a valid Pascal literal
    akString:
      if (Length(A.Default) >= 2) and (A.Default[1] = '"') and (A.Default[Length(A.Default)] = '"') then
      begin
        S := Copy(A.Default, 2, Length(A.Default) - 2);
        R := '';
        I := 1;
        while I <= Length(S) do
        begin
          if (S[I] = '\') and (I < Length(S)) then
          begin
            Inc(I);
            R := R + S[I];
          end
          else
            R := R + S[I];
          Inc(I);
        end;
        Result := QuotedStr(R);
      end;
    akScalarType, akScalarTypeOption:
      if A.Default = '::std::nullopt' then Result := 'kNone'
      else if A.Default = 'at::kByte' then Result := 'kUInt8'
      else if A.Default = 'at::kChar' then Result := 'kInt8'
      else if A.Default = 'at::kShort' then Result := 'kInt16'
      else if A.Default = 'at::kInt' then Result := 'kInt'
      else if A.Default = 'at::kLong' then Result := 'kInt64'
      else if A.Default = 'at::kHalf' then Result := 'kHalf'
      else if A.Default = 'at::kFloat' then Result := 'kFloat'
      else if A.Default = 'at::kDouble' then Result := 'kDouble'
      else if A.Default = 'at::kBool' then Result := 'kBool'
      else if A.Default = 'at::kBFloat16' then Result := 'kBFloat16';
    akLayout, akLayoutOption:
      if A.Default = '::std::nullopt' then Result := 'ltNone'
      else if A.Default = 'at::kStrided' then Result := 'ltStrided'
      else if A.Default = 'at::kSparse' then Result := 'ltSparse';
  end;
  if Result = '' then
    Fail(Op.Name + ': no Pascal default for ' + A.Name + ':' + ArgKindNames[A.Kind] + '=' + A.Default);
end;

{ L3 parameter declaration(s) for one argument. }
function PasParam(const Op: TOp; const A: TArg; Index: Integer): String;
var
  N, D: String;
begin
  if LowerCase(A.Name) = 'self' then
    Fail(Op.Name + ': a non-receiver argument is named self');
  N := Ident(A.Name);
  if Index >= Op.DefaultFrom then
    D := ' = ' + PasDefault(Op, A)
  else
    D := '';
  case A.Kind of
    akTensor, akTensorOption: Result := 'const ' + N + ': TTensor';
    akScalar: Result := 'const ' + N + ': TScalar';
    akBool: Result := N + ': Boolean' + D;
    akInt64: Result := N + ': Int64' + D;
    akDouble: Result := N + ': Double' + D;
    akInt64Option: Result := 'const ' + N + ': TOptInt64';
    akDoubleOption: Result := 'const ' + N + ': TOptDouble';
    akIntList: Result := 'const ' + N + ': array of Int64';
    akIntListOption: Result := 'const ' + N + ': TOptIntList';
    akDoubleList: Result := 'const ' + N + ': array of Double';
    akTensorList, akTensorOptList: Result := 'const ' + N + ': array of TTensor';
    akString: Result := 'const ' + N + ': AnsiString' + D;
    akTensorOptions: Result := Ident(A.Name + '_kind') + ': TKind; const '
      + Ident(A.Name + '_device') + ': TDevice';
    akScalarType, akScalarTypeOption: Result := N + ': TKind' + D;
    akDevice: Result := 'const ' + N + ': TDevice';
    akLayout, akLayoutOption: Result := N + ': TLayout' + D;
  end;
end;

{ L2 parameter declaration(s) for one argument, mirroring gen.ml's
  c_typed_args_list. }
function SysParam(const A: TArg): String;
var
  N: String;

  function Two(const SuffixA, TypeA, SuffixB, TypeB: String): String;
  begin
    Result := Ident(A.Name + SuffixA) + ': ' + TypeA + '; '
      + Ident(A.Name + SuffixB) + ': ' + TypeB;
  end;

begin
  N := Ident(A.Name);
  case A.Kind of
    akTensor, akTensorOption: Result := N + ': PTorchTensor';
    akScalar: Result := N + ': PTorchScalar';
    akBool: Result := N + ': cint';
    akInt64: Result := N + ': Int64';
    akDouble: Result := N + ': Double';
    akInt64Option: Result := Two('_v', 'Int64', '_null', 'Byte');
    akDoubleOption: Result := Two('_v', 'Double', '_null', 'Byte');
    akIntList, akIntListOption: Result := Two('_data', 'PInt64', '_len', 'cint');
    akDoubleList: Result := Two('_data', 'PDouble', '_len', 'cint');
    akTensorList, akTensorOptList: Result := Two('_data', 'PPTorchTensor', '_len', 'cint');
    akString: Result := Two('_ptr', 'PChar', '_len', 'cint');
    akTensorOptions: Result := Two('_kind', 'cint', '_device', 'cint');
    akScalarType, akScalarTypeOption, akDevice: Result := N + ': cint';
    akLayout, akLayoutOption: Result := N + ': ShortInt';
  end;
end;

{ Expression(s) passed to atg_* for one argument. }
function CallArg(const Op: TOp; const A: TArg): String;
var
  N: String;
begin
  if (A.Name = Op.Receiver) and (A.Kind = akTensor) then
    Exit('Self.Raw');
  N := Ident(A.Name);
  case A.Kind of
    akTensor, akTensorOption: Result := N + '.Raw';
    akScalar: Result := A.Name + '__';
    akBool: Result := 'System.Ord(' + N + ')';
    akInt64, akDouble: Result := N;
    akInt64Option, akDoubleOption: Result := N + '.Value, System.Ord(not ' + N + '.HasValue)';
    akIntList: Result := 'System.PInt64(' + N + '), System.Length(' + N + ')';
    akIntListOption: Result := 'TchOptIntListPtr(' + N + '), System.Length(' + N + '.Value)';
    akDoubleList: Result := 'System.PDouble(' + N + '), System.Length(' + N + ')';
    akTensorList, akTensorOptList: Result := 'TchSys.PPTorchTensor(' + A.Name + '__), System.Length(' + A.Name + '__)';
    akString: Result := 'System.PChar(' + N + '), System.Length(' + N + ')';
    akTensorOptions: Result := 'System.Ord(' + Ident(A.Name + '_kind') + '), '
      + Ident(A.Name + '_device') + '.ToC';
    akScalarType, akScalarTypeOption, akLayout, akLayoutOption: Result := 'System.Ord(' + N + ')';
    akDevice: Result := N + '.ToC';
  end;
end;

function IsStatic(const Op: TOp): Boolean;
begin
  Result := Op.Receiver = '-';
end;

function HasOutParams(const Op: TOp): Boolean;
begin
  Result := (Op.Shape = rsFixed) and (Op.NFixed >= 2) and not Op.OutLess;
end;

function IsFunction(const Op: TOp): Boolean;
begin
  Result := (Op.Shape in [rsDynamic, rsBool, rsInt64, rsDouble])
    or ((Op.Shape = rsFixed) and (Op.NFixed = 1));
end;

function ResultType(const Op: TOp): String;
begin
  case Op.Shape of
    rsFixed: Result := 'TTensor';
    rsDynamic: Result := 'TTensorArray';
    rsBool: Result := 'Boolean';
    rsInt64: Result := 'Int64';
    rsDouble: Result := 'Double';
  else
    Result := '';
  end;
end;

{ Joins "Head(" + params + ")Tail" into lines of at most ~80 columns. }
procedure EmitWrapped(L: TStringList; const Indent, Head: String;
  const Params: array of String; const Tail: String);
var
  Line, Piece: String;
  I: Integer;
begin
  Line := Indent + Head;
  if Length(Params) > 0 then
  begin
    Line := Line + '(';
    for I := 0 to High(Params) do
    begin
      Piece := Params[I];
      if I < High(Params) then
        Piece := Piece + ';';
      if (I > 0) and (Length(Line) + 1 + Length(Piece) > 80) then
      begin
        L.Add(Line);
        Line := Indent + '  ' + Piece;
      end
      else if I > 0 then
        Line := Line + ' ' + Piece
      else
        Line := Line + Piece;
    end;
    Line := Line + ')';
  end;
  L.Add(Line + Tail);
end;

function ParamList(const Op: TOp): TStringArray;
var
  I, N: Integer;
  Outs: String;
begin
  Result := nil;
  N := 0;
  for I := 0 to High(Op.Args) do
  begin
    if (Op.Args[I].Name = Op.Receiver) and (Op.Args[I].Kind = akTensor) then
      Continue;
    SetLength(Result, N + 1);
    Result[N] := PasParam(Op, Op.Args[I], I);
    Inc(N);
  end;
  if HasOutParams(Op) then
  begin
    Outs := '';
    for I := 0 to High(Op.Returns) do
    begin
      if I > 0 then
        Outs := Outs + ', ';
      Outs := Outs + Ident(Op.Returns[I]);
    end;
    SetLength(Result, N + 1);
    Result[N] := 'out ' + Outs + ': TTensor';
  end;
end;

procedure EmitIntf(L: TStringList; const Op: TOp);
var
  Head, Tail: String;
begin
  if IsFunction(Op) then
  begin
    Head := 'function ';
    Tail := ': ' + ResultType(Op) + ';';
  end
  else
  begin
    Head := 'procedure ';
    Tail := ';';
  end;
  if IsStatic(Op) then
  begin
    Head := 'class ' + Head;
    Tail := Tail + ' static;';
  end;
  EmitWrapped(L, '    ', Head + Ident(Op.Name), ParamList(Op), Tail);
end;

procedure EmitImpl(L: TStringList; const Op: TOp);
var
  Head, Tail, Call, Args, Ind: String;
  I, NScalars, NLists: Integer;
  A: TArg;
begin
  if IsFunction(Op) then
  begin
    Head := 'function ';
    Tail := ': ' + ResultType(Op) + ';';
  end
  else
  begin
    Head := 'procedure ';
    Tail := ';';
  end;
  if IsStatic(Op) then
    Head := 'class ' + Head;
  EmitWrapped(L, '', Head + 'TTensor.' + Ident(Op.Name), ParamList(Op), Tail);

  NScalars := 0;
  NLists := 0;
  for I := 0 to High(Op.Args) do
    case Op.Args[I].Kind of
      akScalar: Inc(NScalars);
      akTensorList, akTensorOptList: Inc(NLists);
    end;

  { var section }
  if (Op.Shape = rsFixed) or (Op.Shape = rsDynamic) or (Op.Shape in [rsBool, rsInt64, rsDouble])
     or (NScalars + NLists > 0) then
  begin
    L.Add('var');
    case Op.Shape of
      rsFixed: L.Add('  Outs__: array[0..' + IntToStr(Op.NFixed - 1) + '] of PTorchTensor;');
      rsDynamic: L.Add('  P__: PPTorchTensor;');
      rsBool: L.Add('  R__: cint;');
      rsInt64: L.Add('  R__: Int64;');
      rsDouble: L.Add('  R__: Double;');
    end;
    for I := 0 to High(Op.Args) do
      case Op.Args[I].Kind of
        akScalar: L.Add('  ' + Op.Args[I].Name + '__: PTorchScalar;');
        akTensorList, akTensorOptList: L.Add('  ' + Op.Args[I].Name + '__: TTensorPtrArray;');
      end;
  end;
  L.Add('begin');

  { nil-handle checks: receiver first, then the other Tensor arguments }
  if not IsStatic(Op) then
    L.Add('  TchCheckArg(Self, ''' + Op.Receiver + ''', ''' + Op.Name + ''');');
  for I := 0 to High(Op.Args) do
  begin
    A := Op.Args[I];
    if (A.Kind = akTensor) and (A.Name <> Op.Receiver) then
      L.Add('  TchCheckArg(' + Ident(A.Name) + ', ''' + A.Name + ''', ''' + Op.Name + ''');');
  end;
  for I := 0 to High(Op.Args) do
  begin
    A := Op.Args[I];
    case A.Kind of
      akTensorList:
        L.Add('  ' + A.Name + '__ := TchTensorPtrs(' + Ident(A.Name) + ', ''' + A.Name + ''', ''' + Op.Name + ''');');
      akTensorOptList:
        L.Add('  ' + A.Name + '__ := TchTensorOptPtrs(' + Ident(A.Name) + ');');
    end;
  end;

  { scalar temporaries: created inside try so every created one is freed }
  if NScalars > 0 then
  begin
    for I := 0 to High(Op.Args) do
      if Op.Args[I].Kind = akScalar then
        L.Add('  ' + Op.Args[I].Name + '__ := nil;');
    L.Add('  try');
    Ind := '    ';
    for I := 0 to High(Op.Args) do
      if Op.Args[I].Kind = akScalar then
        L.Add(Ind + Op.Args[I].Name + '__ := TchScalarToC(' + Ident(Op.Args[I].Name) + ');');
  end
  else
    Ind := '  ';

  { the call }
  Args := '';
  for I := 0 to High(Op.Args) do
  begin
    if Args <> '' then
      Args := Args + ', ';
    Args := Args + CallArg(Op, Op.Args[I]);
  end;
  case Op.Shape of
    rsFixed:
      begin
        for I := 0 to Op.NFixed - 1 do
          L.Add(Ind + 'Outs__[' + IntToStr(I) + '] := nil;');
        if Args <> '' then
          Args := ', ' + Args;
        L.Add(Ind + 'atg_' + Op.Name + '(@Outs__[0]' + Args + ');');
        L.Add(Ind + 'TchCheckSlots(@Outs__[0], ' + IntToStr(Op.NFixed) + ');');
        if Op.NFixed = 1 then
          L.Add(Ind + 'Result := TTensor.FromRaw(Outs__[0]);')
        else if Op.OutLess then
          L.Add(Ind + 'TchFreeSlots(@Outs__[0], ' + IntToStr(Op.NFixed) + ');  // results are the tensors passed in')
        else
          for I := 0 to Op.NFixed - 1 do
            L.Add(Ind + Ident(Op.Returns[I]) + ' := TTensor.FromRaw(Outs__[' + IntToStr(I) + ']);');
      end;
    rsDynamic:
      begin
        L.Add(Ind + 'P__ := atg_' + Op.Name + '(' + Args + ');');
        L.Add(Ind + 'TchCheck;');
        L.Add(Ind + 'Result := TchTakeTensorArray(P__);');
      end;
    rsBool, rsInt64, rsDouble:
      begin
        L.Add(Ind + 'R__ := atg_' + Op.Name + '(' + Args + ');');
        L.Add(Ind + 'TchCheck;');
        if Op.Shape = rsBool then
          L.Add(Ind + 'Result := R__ <> 0;')
        else
          L.Add(Ind + 'Result := R__;');
      end;
    rsNothing:
      begin
        L.Add(Ind + 'atg_' + Op.Name + '(' + Args + ');');
        L.Add(Ind + 'TchCheck;');
      end;
  end;
  Call := '';
  if NScalars > 0 then
  begin
    L.Add('  finally');
    for I := 0 to High(Op.Args) do
      if Op.Args[I].Kind = akScalar then
        L.Add('    if ' + Op.Args[I].Name + '__ <> nil then ats_free(' + Op.Args[I].Name + '__);');
    L.Add('  end;');
  end;
  L.Add('end;');
  L.Add('');
end;

procedure EmitSys(L: TStringList; const Op: TOp);
var
  Params: TStringArray;
  I, N: Integer;
  Head, Tail: String;
begin
  Params := nil;
  N := 0;
  if Op.Shape = rsFixed then
  begin
    SetLength(Params, 1);
    Params[0] := 'out__: PPTorchTensor';
    N := 1;
  end;
  for I := 0 to High(Op.Args) do
  begin
    SetLength(Params, N + 1);
    Params[N] := SysParam(Op.Args[I]);
    Inc(N);
  end;
  case Op.Shape of
    rsFixed, rsNothing: begin Head := 'procedure '; Tail := ';'; end;
    rsDynamic: begin Head := 'function '; Tail := ': PPTorchTensor;'; end;
    rsBool: begin Head := 'function '; Tail := ': cint;'; end;
    rsInt64: begin Head := 'function '; Tail := ': Int64;'; end;
    rsDouble: begin Head := 'function '; Tail := ': Double;'; end;
  end;
  EmitWrapped(L, '', Head + 'atg_' + Op.Name, Params, Tail + ' cdecl; external LibTch;');
end;

{ True if S consists of routine directives only (`inline;`, `static;
  overload;`): the tail of the previous declaration on a line of its own. }
function IsDirectiveLine(const S: String): Boolean;
var
  Parts: TStringArray;
  W: String;
begin
  Parts := LowerCase(StringReplace(S, ' ', '', [rfReplaceAll])).Split([';']);
  Result := Pos(';', S) > 0;
  for W in Parts do
    if (W <> '') and (W <> 'static') and (W <> 'overload') and (W <> 'inline')
       and (W <> 'virtual') and (W <> 'abstract') and (W <> 'override')
       and (W <> 'reintroduce') then
      Exit(False);
end;

{ A declaration is complete when its parentheses balance and it ends in ';'. }
function DeclComplete(const S: String): Boolean;
var
  C: Char;
  Depth: Integer;
begin
  Depth := 0;
  for C in S do
    if C = '(' then
      Inc(Depth)
    else if C = ')' then
      Dec(Depth);
  Result := (Depth = 0) and (S <> '') and (S[Length(S)] = ';');
end;

{ Member names declared in the TTensor record of Tch.pas (fields, methods,
  properties), lower case, so that generated names and locals can be
  checked against them. Members whose declaration says `overload;` go to
  Overloads instead: deliberate overloads of generated ops are allowed. }
procedure ScanHandWritten(const FileName: String);
var
  Lines: TStringList;
  I, P: Integer;
  S, W, Decl, LastName: String;
  InRecord, InComment, LastOverload: Boolean;

  procedure Commit;
  begin
    if LastName = '' then
      Exit;
    if LastOverload then
      Overloads.Add(LastName)
    else
      HandWritten.Add(LastName);
    LastName := '';
  end;

begin
  InComment := False;
  HandWritten := TStringList.Create;
  HandWritten.Sorted := True;
  HandWritten.Duplicates := dupIgnore;
  Overloads := TStringList.Create;
  Overloads.Sorted := True;
  Overloads.Duplicates := dupIgnore;
  Lines := TStringList.Create;
  try
    Lines.LoadFromFile(FileName);
    InRecord := False;
    Decl := '';
    LastName := '';
    LastOverload := False;
    for I := 0 to Lines.Count - 1 do
    begin
      S := Trim(Lines[I]);
      if not InRecord then
      begin
        if Pos('TTensor = record', S) = 1 then
          InRecord := True;
        Continue;
      end;
      if S = 'end;' then
        Break;
      if (Pos('{$', S) = 1) or (Pos('//', S) = 1) or (S = '') then
        Continue;
      if InComment then
      begin
        if Pos('}', S) > 0 then
          InComment := False;
        Continue;
      end;
      if S[1] = '{' then
      begin
        if Pos('}', S) = 0 then
          InComment := True;
        Continue;
      end;
      { continuation of a multi-line declaration, or its directives alone }
      if (Decl <> '') or IsDirectiveLine(S) then
      begin
        Decl := Decl + ' ' + S;
        if Pos('overload;', LowerCase(S)) > 0 then
          LastOverload := True;
        if DeclComplete(Decl) then
          Decl := '';
        Continue;
      end;
      Commit;
      W := '';
      if Pos('class function ', S) = 1 then W := Copy(S, 16, MaxInt)
      else if Pos('class procedure ', S) = 1 then W := Copy(S, 17, MaxInt)
      else if Pos('function ', S) = 1 then W := Copy(S, 10, MaxInt)
      else if Pos('procedure ', S) = 1 then W := Copy(S, 11, MaxInt)
      else if Pos('property ', S) = 1 then W := Copy(S, 10, MaxInt)
      else if Pos('class operator ', S) = 1 then
        W := ''   // no name, but may continue on the next lines
      else if (S = 'private') or (S = 'public') or (S = 'public type')
        or (S = 'strict private') or (S = 'protected') then
        Continue
      else
        W := S;   // a field (name: Type;) or a nested type (Name = ...;)
      P := 1;
      while (P <= Length(W)) and (W[P] in ['&', 'A'..'Z', 'a'..'z', '0'..'9', '_']) do
        Inc(P);
      W := Copy(W, 1, P - 1);
      if (W <> '') and (W[1] = '&') then
        Delete(W, 1, 1);
      LastName := LowerCase(W);
      LastOverload := Pos('overload;', LowerCase(S)) > 0;
      if not DeclComplete(S) then
        Decl := S;
    end;
    Commit;
    if not InRecord then
      Fail('TTensor = record not found in ' + FileName);
  finally
    Lines.Free;
  end;
end;

procedure CheckNames;
var
  I, J: Integer;
  N: String;
begin
  OpNames := TStringList.Create;
  OpNames.Sorted := True;
  OpNames.Duplicates := dupError;
  for I := 0 to High(Ops) do
  begin
    N := LowerCase(Ops[I].Name);
    if HandWritten.IndexOf(N) >= 0 then
      Fail('op ' + Ops[I].Name + ' collides with a hand-written TTensor member');
    try
      OpNames.Add(N);
    except
      Fail('op name ' + Ops[I].Name + ' is not unique (case-insensitively)');
    end;
  end;
  { generated locals must not be members; parameters must not shadow the
    unit names and helpers the bodies refer to (intrinsics are written
    System.Ord etc. because parameters named ord and length exist) }
  for I := 0 to High(Ops) do
  begin
    for J := 0 to High(Ops[I].Args) do
    begin
      N := LowerCase(Ops[I].Args[J].Name + '__');
      if (HandWritten.IndexOf(N) >= 0) or (Overloads.IndexOf(N) >= 0)
         or (OpNames.IndexOf(N) >= 0) then
        Fail('local ' + N + ' in ' + Ops[I].Name + ' collides with a member');
      N := LowerCase(Ops[I].Args[J].Name);
      if Ops[I].Args[J].Name = Ops[I].Receiver then
        Continue;
      if (N = 'system') or (N = 'tchsys') or (N = 'tch') or (N = 'ttensor')
         or (N = 'self') or (Copy(N, 1, 3) = 'tch') then
        Fail('parameter ' + N + ' in ' + Ops[I].Name + ' would shadow a name used by generated bodies');
    end;
    for N in ['outs__', 'p__', 'r__'] do
      if (HandWritten.IndexOf(N) >= 0) or (Overloads.IndexOf(N) >= 0)
         or (OpNames.IndexOf(N) >= 0) then
        Fail('local ' + N + ' collides with a member');
  end;
end;

procedure Generate;
var
  Sys, Intf, Impl: TStringList;
  I, NDefaults, NOutLess, NOuts: Integer;
begin
  Sys := TStringList.Create;
  Intf := TStringList.Create;
  Impl := TStringList.Create;
  try
    Sys.Add('{ THIS FILE IS AUTOMATICALLY GENERATED by gen/tchgen.pas, DO NOT EDIT BY HAND!');
    Sys.Add('  L2: external declarations of every atg_* function in libtch');
    Sys.Add('  (csrc/libtch/torch_api_generated.h). Types per DESIGN.md section 2. }');
    Sys.Add('unit TchSysGenerated;');
    Sys.Add('');
    Sys.Add('{$mode objfpc}{$H+}');
    Sys.Add('');
    Sys.Add('interface');
    Sys.Add('');
    Sys.Add('uses');
    Sys.Add('  ctypes, TchSys;');
    Sys.Add('');
    Intf.Add('{ THIS FILE IS AUTOMATICALLY GENERATED by gen/tchgen.pas, DO NOT EDIT BY HAND!');
    Intf.Add('  Generated op methods of TTensor (DESIGN.md 4.4); included inside the');
    Intf.Add('  record declaration in Tch.pas. }');
    Impl.Add('{ THIS FILE IS AUTOMATICALLY GENERATED by gen/tchgen.pas, DO NOT EDIT BY HAND!');
    Impl.Add('  Bodies of the generated op methods; included in the implementation');
    Impl.Add('  section of Tch.pas. }');
    Impl.Add('');
    NDefaults := 0;
    NOutLess := 0;
    NOuts := 0;
    for I := 0 to High(Ops) do
    begin
      EmitSys(Sys, Ops[I]);
      EmitIntf(Intf, Ops[I]);
      EmitImpl(Impl, Ops[I]);
      if Ops[I].DefaultFrom < Length(Ops[I].Args) then
        Inc(NDefaults);
      if Ops[I].OutLess then
        Inc(NOutLess);
      if HasOutParams(Ops[I]) then
        Inc(NOuts);
    end;
    Sys.Add('');
    Sys.Add('implementation');
    Sys.Add('');
    Sys.Add('end.');
    Sys.SaveToFile(SrcDir + '/TchSysGenerated.pas');
    Intf.SaveToFile(SrcDir + '/TchTensorGenerated_intf.inc');
    Impl.SaveToFile(SrcDir + '/TchTensorGenerated_impl.inc');
    WriteLn('tchgen: ', Length(Ops), ' ops; ', NDefaults, ' with default parameters, ',
      NOuts, ' with out parameters, ', NOutLess, ' fill tensors passed in');
  finally
    Sys.Free;
    Intf.Free;
    Impl.Free;
  end;
end;

var
  I: Integer;
begin
  OpsTsv := 'gen/ops.tsv';
  SrcDir := 'src';
  if ParamCount >= 1 then
    OpsTsv := ParamStr(1);
  if ParamCount >= 2 then
    SrcDir := ParamStr(2);
  ReadOps(OpsTsv);
  for I := 0 to High(Ops) do
  begin
    ClassifyOutLess(Ops[I]);
    ComputeDefaultRun(Ops[I]);
  end;
  ScanHandWritten(SrcDir + '/Tch.pas');
  CheckNames;
  Generate;
end.
