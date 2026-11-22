# 詳細設計書 — unit `Recompute` (T13)

## 1. モジュール構成

### 1.1 ソースファイル

| ファイル | 内容 |
|---|---|
| `src/Recompute.pas` | unit Recompute: ブロック単位再計算バックワード |

### 1.2 クラス・関数一覧

| 名前 | 種類 | 責務 |
|---|---|
| `ERecompute` | 例外クラス | Recompute 処理中のエラー |
| `TStage` | 抽象クラス | ステージのインタフェース (Device, Forward) |
| `TStageArray` | 型 | ステージの動的配列 |
| `TSavedInputs` | クラス | ステージ入力テンソルの保存・管理 |
| `RecomputeForward` | 関数 | 全ステージをグラフなしで実行し入力を保存 |
| `RecomputeBackward` | 関数 | 保存入力から再実行し勾配を伝播 |

## 2. 関数仕様

### 2.1 ERecompute

```pascal
ERecompute = class(Exception);
```

Recompute 処理中のエラーを表す例外クラス。`CreateFmt` でメッセージを整形して送出する。

### 2.2 TStage

```pascal
TStage = class abstract
public
  function Device: TDevice; virtual; abstract;
  function Forward(const X: TTensor): TTensor; virtual; abstract;
end;
```

| メソッド | 引数 | 戻り値 | 説明 |
|---|---|---|---|
| Device | なし | TDevice | ステージが実行されるデバイス |
| Forward | X: TTensor | TTensor | 入力 X に対するステージの順伝播 |

### 2.3 TStageArray

```pascal
TStageArray = array of TStage;
```

ステージの動的配列。`nil` または空配列の場合、RecomputeForward はエラーとする。

### 2.4 TSavedInputs

```pascal
TSavedInputs = class
public
  constructor Create(AOnHost: Boolean);
  procedure Clear;
  procedure Add(const X: TTensor);
  function Count: Integer;
  function Item(I: Integer): TTensor;
  property OnHost: Boolean read FOnHost;
end;
```

| メソッド | 引数 | 戻り値 | 事前条件 | 説明 |
|---|---|---|---|---|
| Create | AOnHost: Boolean | TSavedInputs | なし | OnHost フラグを設定し、空のリストを初期化 |
| Clear | なし | なし | なし | 保存内容をクリア |
| Add | X: TTensor | なし | なし | X を detach し、OnHost の場合は CPU へ転送して保存 |
| Count | なし | Integer | なし | 保存されたテンソル数 |
| Item | I: Integer | TTensor | 0 ≤ I < Count | I 番目の保存テンソルを返す |
| OnHost | なし | Boolean | なし | コンストラクタで設定されたフラグ |

**例外:**

- `Item(I)`: I < 0 または I >= Count の場合、`ERecompute` を送出
  - メッセージ: `'saved input %d out of range (count %d)'`

### 2.5 RecomputeForward

```pascal
function RecomputeForward(const Stages: TStageArray; const X0: TTensor;
  Saved: TSavedInputs): TTensor;
```

| 引数 | 型 | 方向 | 説明 |
|---|---|---|---|
| Stages | TStageArray | 入力 | ステージの配列 |
| X0 | TTensor | 入力 | 初期入力 |
| Saved | TSavedInputs | 出力 | 各ステージの入力を保存 |

| 戻り値 | 型 | 説明 |
|---|---|---|
| 結果 | TTensor | 最終ステージの出力 (RequiresGrad=False) |

**事前条件:**

1. Stages が `nil` または空でないこと
2. X0 が defined (undefined でない) こと

**例外:**

1. Stages が `nil` または空: `ERecompute('no stages')`
2. X0 が undefined: `ERecompute('X0 is undefined')`

### 2.6 RecomputeBackward

```pascal
function RecomputeBackward(const Stages: TStageArray; Saved: TSavedInputs;
  const GradOut: TTensor): TTensor;
```

| 引数 | 型 | 方向 | 説明 |
|---|---|---|---|
| Stages | TStageArray | 入力 | RecomputeForward と同じ配列 |
| Saved | TSavedInputs | 入力 | Forward で保存された入力 |
| GradOut | TTensor | 入力 | 出力に対する勾配 |

| 戻り値 | 型 | 説明 |
|---|---|---|
| 結果 | TTensor | 入力 X0 に対する勾配 dL/dX0 (最初のステージのデバイス上) |

**事前条件:**

1. Saved.Count = Length(Stages) であること
2. GradOut が defined であること

**例外:**

1. Saved.Count <> Length(Stages): `ERecompute.CreateFmt('saved inputs: %d, stages: %d', ...)`
2. GradOut が undefined: `ERecompute('GradOut is undefined')`
3. ループ内で Y と G の形状が不一致: `ERecompute.CreateFmt('stage %d: output %s, gradient %s', ...)`

## 3. データ構造

### 3.1 TSavedInputs 内部構造

```pascal
TSavedInputs = class
private
  FOnHost: Boolean;           // CPU へ転送するかどうか
  FItems: array of TTensor;   // 保存されたテンソルの動的配列
public
  // ...
end;
```

| フィールド | 型 | 説明 |
|---|---|---|
| FOnHost | Boolean | Create で設定。True の場合、Add で CPU へ転送 |
| FItems | array of TTensor | 保存されたテンソル。TTensor は管理レコードなので Free 不要 |

### 3.2 TStageArray

```pascal
TStageArray = array of TStage;
```

ステージへの参照を格納する動的配列。ステージの所有権は呼び出し元にある。

## 4. 処理詳細

### 4.1 TSavedInputs.Create

```
1. inherited Create を呼び出す
2. FOnHost := AOnHost
```

### 4.2 TSavedInputs.Clear

```
1. SetLength(FItems, 0)
```

FItems は TTensor の動的配列。TTensor は管理レコード (reference counted) なので、
配列を縮小すると自動的に参照が解放される。

### 4.3 TSavedInputs.Add

```
1. N := Length(FItems)
2. SetLength(FItems, N + 1)
3. FOnHost の場合:
     FItems[N] := X.detach.&to(TDevice.Cpu)
   そうでない場合:
     FItems[N] := X.detach
```

- `detach`: グラフから切り離す (grad_fn を持たない新しいテンソル)
- `&to`: reserved word `to` をエスケープ
- `TDevice.Cpu`: CPU デバイスへ転送

### 4.4 TSavedInputs.Count

```
1. Result := Length(FItems)
```

### 4.5 TSavedInputs.Item

```
1. I < 0 または I >= Count の場合:
     raise ERecompute.CreateFmt('saved input %d out of range (count %d)', [I, Count])
2. Result := FItems[I]
```

### 4.6 RecomputeForward

```
1. Stages が nil または空 (Length(Stages) = 0) の場合:
     raise ERecompute.Create('no stages')
2. X0 が undefined (not X0.Defined) の場合:
     raise ERecompute.Create('X0 is undefined')
3. Saved.Clear
4. G := NoGradGuard (G: IInterface; ローカル変数)
5. X := X0
6. I := 0 から High(Stages) まで:
     a. X := X.&to(Stages[I].Device)
     b. Saved.Add(X)
     c. X := Stages[I].Forward(X)
7. Result := X
```

- NoGradGuard はスコープ終了時に自動的にグラフ記録を再有効化する (IInterface)
- 各ステージの Forward はグラフを作らない (LastRequiresGrad = False)
- 出力は RequiresGrad = False

### 4.7 RecomputeBackward

```
1. Saved.Count <> Length(Stages) の場合:
     raise ERecompute.CreateFmt('saved inputs: %d, stages: %d', [Saved.Count, Length(Stages)])
2. GradOut が undefined (not GradOut.Defined) の場合:
     raise ERecompute.Create('GradOut is undefined')
3. G := GradOut
4. I := High(Stages) downto 0:
     a. X := Saved.Item(I).&to(Stages[I].Device).detach.requires_grad_(True)
     b. Y := Stages[I].Forward(X)
     c. G := G.&to(Stages[I].Device)
     d. ShapeStr(Y) <> ShapeStr(G) の場合:
          raise ERecompute.CreateFmt('stage %d: output %s, gradient %s',
            [I, ShapeStr(Y), ShapeStr(G)])
     e. (Y * G).sum.Backward
     f. G := X.grad
5. Result := G
```

- 各ステージでグラフを作成・破棄 (LastRequiresGrad = True)
- `(Y * G).sum.Backward`: 要素積の和で勾配を伝播
- `X.grad`: requires_grad_(True) で設定した入力の勾配
- X, Y, G は TTensor のローカル変数 (reference counted)

### 4.8 補助関数 ShapeStr (private, 実装セクション)

```pascal
function ShapeStr(const T: TTensor): string;
```

テンソルの形状を文字列で返す (例: `'[2, 3]'`)。

## 5. 変更履歴

| 日付 | 変更内容 | 理由 |
|---|---|---|
| 2024-11-22 | 初版作成 | T13 文書化作業 |
