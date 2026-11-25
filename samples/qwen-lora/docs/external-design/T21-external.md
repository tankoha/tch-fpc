# T21 外部設計書

## 1. 概要

**目的**: 学習・推論の計算精度を BF16 に変更し、メモリ使用量を削減して速度を向上させる。

**利用者**: harness owner (測定実験のみ) および Qwen モデルを使う開発者。

**前提**: 
- 2 枚の GPU と CPU で動作する qlora_train, qlora_logprobs が既にある (T11, T17)。
- tch-fpc が BF16 型をサポートしている。

## 2. 機能一覧

| 機能 ID | 名称 | 概要 |
|---|---|---|
| T21-F01 | BF16 計算モード | トランクブロックの行列積を BF16 で計算 |
| T21-F02 | CPU でのアダプタ初期値生成 | 再現性のためアダプタ初期値を CPU で生成 |

## 3. 入出力

### 3.1 コマンドラインオプション

**プログラム**: qlora_train, qlora_logprobs

| 項目名 | 型 | 桁 | 必須 | デフォルト | チェック仕様 |
|---|---|---|---|---|---|
| --compute | 文字列 | 4 文字 | いいえ | fp32 | "fp32" または "bf16" (大文字小文字区別、完全一致) |

### 3.2 設定ハッシュ

`--compute` は設定ハッシュに含まれない。チェックポイントは fp32 で保存し、bf16 で再開できる (逆も可)。

## 4. 外部インタフェース

なし (既存のモデルファイル、チェックポイントファイルを変更しない)。

## 5. 処理概要

### 5.1 BF16 計算モード (T21-F01)

- `--compute bf16` 指定時、以下の行列積を BF16 で実行:
  - トランクブロック内のすべての TPackedLinear (FFN、QKV、attention gate、SSM)
  - 入力 X (fp32) と重みを BF16 に変換、行列積を計算、結果を fp32 に戻す
  - LoRA の Delta も BF16 変換後の X を使用

- 以下は fp32 のまま:
  - 残差ストリーム
  - RMS norm
  - RoPE
  - 畳み込み
  - delta rule
  - attention (QKV 射影以外)
  - ヘッド (logits 計算)
  - 損失計算

設計理由:
- ヘッドはチャンクごとに dequantise するため、fp32 でもメモリコストが低い
- BF16 logits は有効数字 3 桁 (0.0625 刻み) で、語彙選択に影響する可能性

### 5.2 CPU でのアダプタ初期値生成 (T21-F02)

- `TLoraPair.Create` で lora_a を CPU で生成:
  - CPU の Generator で zeros → uniform
  - VarCopy で GPU メモリに転送

設計理由:
- デバイスマップに関わらず同じ初期値 (再現性)
- 測定 M2g-a で、GPU ごとの生成は初期値が異なり勾配が変化 (3.44 vs 3.77 at update 1)

### 5.3 デフォルト動作

- `--compute` 未指定時は fp32 (従来の動作、ビットレベルで同一)
- 既存の UT、ITa、オラクルはすべて変更なし

## 6. エラー・例外

| エラー ID | 発生源 | メッセージ | 利用者への見え方 |
|---|---|---|---|
| E21-01 | qlora_train --compute | --compute: fp32 or bf16, got "<value>" | 標準エラー出力、終了コード 1 |
| E21-02 | qlora_train --compute | --compute: value missing | 標準エラー出力、終了コード 1 |
| E21-03 | qlora_logprobs --compute | --compute needs a value | 標準エラー出力、終了コード 2 |
| E21-04 | qlora_logprobs --compute | --compute must be fp32 or bf16, got <value> | 標準エラー出力、終了コード 2 |

値の比較は完全一致 (大文字小文字区別、"BF16" は拒否)。

## 7. 非機能

### 7.1 性能

測定結果 (プロトタイプ、2 枚 GPU):
- fp32 vs CPU 曲線: 差 5.5e-3 以下 (16 updates)
- bf16 vs CPU 曲線: 6.3% 以下 (update 9 まで 2.2e-3)、最終 2.9%
- 学習時間: bf16 で 158s vs fp32 219s (約 28% 短縮)

### 7.2 UT

- CPU で BF16 の UT を実行 (GPU 不要)
- 全 UT 行 100% パス (5 プログラム、行カバレッジ 100%)

### 7.3 ITa

- CPU で bf16 ITa を実行 (約 7 分)
- fp32 オラクルとの差異の限界値:
  - logprobs: 0.25 以下
  - loss: 相対差 2% 以下

## 8. 変更履歴

| 日付 | 変更内容 | 理由 |
|---|---|---|
| 2025-11-25 | 初版作成 | T21a (BF16 計算追加) |
