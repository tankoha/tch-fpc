# External design — unit `Recompute` (T13)

## 1. 概要

ブロック単位の再計算によるバックワード (design 8.14)。フォワードはグラフを
作らず各ステージの入力を保存し、バックワードは最終ステージから順に各ステージ
を再実行して勾配を伝播する。パラメータ勾配は各ステージの葉に蓄積される。

Qwen には依存しない。ステージとは Device と Forward を持つ任意のオブジェクト。
T17 は 64 個のトランクブロックを 2 つの GPU で実行するのにこれを使う。

## 2. 機能一覧

| 機能 ID | 名称 | 概要 |
|---|---|---|
| T13-01 | TSavedInputs | ステージ入力テンソルの保存と取り出し |
| T13-02 | RecomputeForward | 全ステージをグラフなしで実行し、各入力を保存 |
| T13-03 | RecomputeBackward | 保存された入力から各ステージを再実行し、勾配を伝播 |

## 3. 入出力

### T13-01 TSavedInputs

| 操作 | 引数 | 戻り値 | 備考 |
|---|---|---|---|
| Create | OnHost: Boolean | TSavedInputs | OnHost=True のとき、追加時に CPU へ転送 |
| Clear | なし | なし | 保存内容をクリア |
| Add | X: TTensor | なし | X を detach して保存 (OnHost なら CPU へ) |
| Count | なし | Integer | 保存されたテンソル数 |
| Item | I: Integer | TTensor | I 番目の保存テンソル (0 ベース) |
| OnHost | なし | Boolean | コンストラクタで設定されたフラグ |

### T13-02 RecomputeForward

| 引数 | 型 | 方向 | 説明 |
|---|---|---|---|
| Stages | TStageArray | 入力 | ステージの配列 (空不可) |
| X0 | TTensor | 入力 | 初期入力 (undefined 不可) |
| Saved | TSavedInputs | 出力 | 各ステージの入力を保存 |

| 戻り値 | 型 | 説明 |
|---|---|---|
| 結果 | TTensor | 最終ステージの出力 (RequiresGrad=False) |

### T13-03 RecomputeBackward

| 引数 | 型 | 方向 | 説明 |
|---|---|---|---|
| Stages | TStageArray | 入力 | RecomputeForward と同じ配列 |
| Saved | TSavedInputs | 入力 | Forward で保存された入力 |
| GradOut | TTensor | 入力 | 出力に対する勾配 (undefined 不可) |

| 戻り値 | 型 | 説明 |
|---|---|---|
| 結果 | TTensor | 入力 X0 に対する勾配 dL/dX0 (最初のステージのデバイス上) |

## 4. 外部インタフェース

TStage 抽象クラスを通じて呼び出し元が提供するステージを使用する。

```pascal
TStage = class abstract
public
  function Device: TDevice; virtual; abstract;
  function Forward(const X: TTensor): TTensor; virtual; abstract;
end;
```

## 5. 処理概要

### T13-01 TSavedInputs

- Create: OnHost フラグを設定
- Add: テンソルを detach し、OnHost の場合は CPU へ転送して保存
- Item: インデックス範囲チェック後、テンソルを返す

### T13-02 RecomputeForward

1. Saved をクリア
2. NoGradGuard 下で、各ステージを順に実行
3. 各ステージの入力を Saved に追加
4. 最終出力を返す

### T13-03 RecomputeBackward

1. Saved.Count と Stages の長さが一致することを確認
2. 最終ステージから最初のステージへ逆順に:
   - 保存された入力を取り出し、requires_grad=True で再実行
   - 勾配を受け取り、(Y * G).sum.Backward で勾配を伝播
   - 入力勾配を次のステージへ
3. 最初のステージの入力勾配を返す

## 6. エラー・例外

ERecompute 例外を送出する。

| 起点条件 | メッセージ | 例 |
|---|---|---|
| TSavedInputs.Item(I): I < 0 または I >= Count | saved input %d out of range (count %d) | saved input 2 out of range (count 2) |
| RecomputeForward: Stages が空 | no stages | - |
| RecomputeForward: X0 が undefined | X0 is undefined | - |
| RecomputeBackward: Saved.Count <> Length(Stages) | saved inputs: %d, stages: %d | saved inputs: 1, stages: 2 |
| RecomputeBackward: GradOut が undefined | GradOut is undefined | - |
| RecomputeBackward: ループ内で形状不一致 | stage %d: output %s, gradient %s | stage 1: output [2], gradient [3] |

## 7. 非機能

- 正確性: fp64 で通常のバックワードとビット一致 (差分 0)
- 性能: ステージ数 N に対し、フォワード 1 回 + バックワード 1 回で計算量 O(N)
- メモリ: バックワード時、1 ステージ分のグラフのみを保持

## 8. 変更履歴

| 日付 | 変更内容 | 理由 |
|---|---|---|
| 2024-11-22 | 初版作成 | T13a 文書化作業 |
