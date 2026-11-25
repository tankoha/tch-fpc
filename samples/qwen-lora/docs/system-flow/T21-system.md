# T21 システムフロー

## 利用者とジョブ

本タスクの機能は以下の測定・検証ジョブで使用される:

### M2g-b: 2枚カードの BF16 分割検証

```mermaid
flowchart LR
    A[harness owner] --> B[T21-run.sh]
    B --> C[qlora_train]
    B --> D[qlora_logprobs]
    C --> E[CPU オラクル曲線]
    D --> F[T11 プロンプトセット]
    C --> G[work/ita/T21/summary.tsv]
    D --> G
```

- **目的**: BF16 分割の精度・速度を CPU 学習曲線と比較
- **比較対象**: FP32 2枚分割 (M2g-a) の結果
- **実施者**: harness owner
- **環境**: 2枚 GPU、CPU

### M4: 27B モデル検証 (H43)

```mermaid
flowchart LR
    A[harness owner] --> B[qlora_train --compute bf16]
    B --> C[27B モデル]
    B --> D[H43 チェック]
    D --> E[結果レポート]
```

- **目的**: Qwen3.8 27B の BF16 学習を検証
- **環境**: H43 (2 GPU)

### UT: 単体テスト

```mermaid
flowchart LR
    A[UT 実行] --> B[CPU 上で BF16 計算]
    B --> C[5 プログラム]
    C --> D[行カバレッジ 100%]
```

- **環境**: CPU、stock fpc (GPU 非表示)
- **対象**: PackedLinear, Lora, QwenModel, TrainCli, LogprobsCli

### ITa: 結合テスト

```mermaid
flowchart LR
    A[ITa runner] --> B[qlora_logprobs]
    A --> C[qlora_train]
    B --> D[T11 プロンプト]
    C --> E[T17 学習記録]
    D --> F[CPU fp32 オラクル]
    E --> F
    B --> G[summary.tsv]
    C --> G
```

- **実行**: `QLORA_MODEL=<rehearsal GGUF> <project root>/tests/ita/T21-run.sh`
- **結果**: `<project root>/work/ita/T21/summary.tsv`
- **オラクル**: T21 以前のコードの fp32 出力
- **環境**: CPU のみ
