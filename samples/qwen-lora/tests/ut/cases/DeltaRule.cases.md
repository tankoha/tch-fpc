# UT cases — unit `DeltaRule` (T08)

| Case ID | Title | Target function | Input (short) | Expected result | Notes |
|---|---|---|---|---|---|
| UT-DeltaRule-001 | Hand case, token 0 | DeltaStep | M zeros([1, 2, 2]), Q Make([1, 1], [1, 2]), K Make([1, 0], [1, 2]), V Make([2, 4], [1, 2]), Beta Make([0.5], [1]), G Make([0], [1]) | result `[1, 2, 2]` `1 0 2 0`; O `[1, 2]` `1 2` | |
| UT-DeltaRule-002 | Hand case, token 1 | DeltaStep | M Make([1, 0, 2, 0], [1, 2, 2]), Q (1, 2), K (1, 1), V (3, 3), Beta 0.5, G 0 (shapes as 001) | result `2 1 2.5 0.5`; O `4 3.5` | |
| UT-DeltaRule-003 | Hand case, token 1 with decay | DeltaStep | as 002 with G Make([Ln(0.5)], [1]) | result `1.75 1.25 2 1`; O `4.25 4` | |
| UT-DeltaRule-004 | Two heads | DeltaStep | M Make([0, 0, 0, 0, 1, 0, 2, 0], [2, 2, 2]), Q Make([1, 1, 1, 2], [2, 2]), K Make([1, 0, 1, 1], [2, 2]), V Make([2, 4, 3, 3], [2, 2]), Beta Make([0.5, 0.5], [2]), G Make([0, 0], [2]) | result `[2, 2, 2]` `1 0 2 0 2 1 2.5 0.5`; O `[2, 2]` `1 2 4 3.5` | |
| UT-DeltaRule-005 | β = 0, random inputs | DeltaStep | `RandomInputs(23, 3, 1, 4, 5, 1.0, …)`, then M := `TTensor.randn([3, 5, 4])`; step with token 0 (`select(1, 0)`) of Q, K, V, G and Beta zeros([3]) | exact: result = `M * G.select(1, 0).exp.view([3, 1, 1])`; O = `result.matmul(Q.select(1, 0).unsqueeze(2)).squeeze_dim(2)` | |
| UT-DeltaRule-006 | G = 0, random inputs | DeltaStep | as 005 with G zeros([3]) too | exact: result = M | |
| UT-DeltaRule-007 | One step, random inputs vs RefTokens | DeltaStep | `RandomInputs(21, 3, 1, 4, 5, 1.0, …)`; step from zeros([3, 5, 4]) with token 0; `RefTokens` of the same inputs | O vs `RO.select(1, 0)` and result vs RS ≤ 1e-6 (measured 1.5e-8, 3.0e-8) | |
| UT-DeltaRule-008 | Two steps vs RefTokens | DeltaStep | `RandomInputs(22, 3, 2, 4, 5, 1.0, …)`; two steps from zeros([3, 5, 4]) (tokens 0, 1); `RefTokens` | second O vs `RO.select(1, 1)` and result vs RS ≤ 1e-6 (measured 6.0e-8, 1.2e-7) | |
| UT-DeltaRule-009 | M wrong kind | DeltaStep | M zeros([2, 5, 4], kDouble) | `M must be kFloat (fp32)` | |
| UT-DeltaRule-010 | G wrong kind | DeltaStep | G zeros([2], kBFloat16) | `G must be kFloat (fp32)` | |
| UT-DeltaRule-011 | M not 3-D | DeltaStep | M zeros([5, 4]) | `M must be [H, Dv, Dk], got [5, 4]` | |
| UT-DeltaRule-012 | Q wrong shape | DeltaStep | Q zeros([2, 3]) | `Q must be [2, 4], got [2, 3]` | |
| UT-DeltaRule-013 | K wrong head count | DeltaStep | K zeros([1, 4]) | `K must be [2, 4], got [1, 4]` | |
| UT-DeltaRule-014 | V wrong Dv | DeltaStep | V zeros([2, 4]) | `V must be [2, 5], got [2, 4]` | |
| UT-DeltaRule-015 | Beta wrong shape | DeltaStep | Beta zeros([3]) | `Beta must be [2], got [3]` | |
| UT-DeltaRule-016 | G not 1-D | DeltaStep | G zeros([2, 1]) | `G must be [2], got [2, 1]` | |
| UT-DeltaRule-017 | Q wrong kind | DeltaStep | Q zeros([2, 4], kDouble) | `Q must be kFloat (fp32)` | |
| UT-DeltaRule-018 | K wrong kind | DeltaStep | K zeros([2, 4], kDouble) | `K must be kFloat (fp32)` | |
| UT-DeltaRule-019 | V wrong kind | DeltaStep | V zeros([2, 5], kDouble) | `V must be kFloat (fp32)` | |
| UT-DeltaRule-020 | Beta wrong kind | DeltaStep | Beta zeros([2], kDouble) | `Beta must be kFloat (fp32)` | |
| UT-DeltaRule-101 | Hand case, two tokens | DeltaRuleTokens | Q Make([1, 1, 1, 2], [1, 2, 2]), K Make([1, 0, 1, 1], [1, 2, 2]), V Make([2, 4, 3, 3], [1, 2, 2]), Beta Make([0.5, 0.5], [1, 2]), G Make([0, 0], [1, 2]) | O `[1, 2, 2]` `1 2 4 3.5`; State `[1, 2, 2]` `2 1 2.5 0.5` | |
| UT-DeltaRule-102 | Hand case with decay on token 1 | DeltaRuleTokens | as 101 with G Make([0, Ln(0.5)], [1, 2]) | O `1 2 4.25 4`; State `1.75 1.25 2 1` | |
| UT-DeltaRule-103 | β = 0, zero state | DeltaRuleTokens | `RandomInputs(7, 3, 5, 4, 6, 1.0, …)`, Beta replaced by zeros([3, 5]) | O `[3, 5, 6]`, State `[3, 6, 4]`; `O.abs.max` and `State.abs.max` are 0 (the acceptance: the state stays zero) | |
| UT-DeltaRule-104 | g = −inf, each token sees only itself | DeltaRuleTokens | `RandomInputs(8, 3, 5, 4, 6, 1.0, …)`, G replaced by `TTensor.zeros([3, 5]) + Double(NegInfinity)` (unit `Math`) | O vs `(V * Beta.unsqueeze(2)) * (K * Q).sum_dim_intlist(IntList([2]), True, kNone)` ≤ 1e-6 (measured 6.0e-8): each token sees only itself | |
| UT-DeltaRule-105 | T = 1 equals one step | DeltaRuleTokens | `RandomInputs(9, 3, 5, 4, 6, 1.0, …)`; `DeltaRuleTokens` of `narrow(1, 0, 1)` of all five; `DeltaStep` from zeros([3, 6, 4]) with `select(1, 0)` | exact: O `select(1, 0)` = the step's O; State = the step's result | |
| UT-DeltaRule-106 | Heads are independent | DeltaRuleTokens | inputs of 105, whole; and `narrow(0, 1, 1)` of all five | exact: `O.narrow(0, 1, 1)` and `State.narrow(0, 1, 1)` equal the single head's O and State | |
| UT-DeltaRule-107 | Random, T = 20 vs RefTokens | DeltaRuleTokens | `RandomInputs(11, 3, 20, 4, 5, 0.1, …)`; `RefTokens` | O vs RO and State vs RS ≤ 1e-5 (measured 4.8e-7, 1.2e-7) | |
| UT-DeltaRule-108 | Random, larger sizes vs RefTokens | DeltaRuleTokens | `RandomInputs(12, 2, 65, 16, 16, 5.0, …)`; `RefTokens` | ≤ 1e-5 (measured 2.4e-7, 3.0e-8) | |
| UT-DeltaRule-109 | Q wrong kind | DeltaRuleTokens | Q zeros([2, 3, 4], kDouble) | `Q must be kFloat (fp32)` | |
| UT-DeltaRule-110 | G wrong kind | DeltaRuleTokens | G zeros([2, 3], kBFloat16) | `G must be kFloat (fp32)` | |
| UT-DeltaRule-118 | K wrong kind | DeltaRuleTokens | K zeros([2, 3, 4], kDouble) | `K must be kFloat (fp32)` | |
| UT-DeltaRule-119 | V wrong kind | DeltaRuleTokens | V zeros([2, 3, 5], kDouble) | `V must be kFloat (fp32)` | |
| UT-DeltaRule-120 | Beta wrong kind | DeltaRuleTokens | Beta zeros([2, 3], kDouble) | `Beta must be kFloat (fp32)` | |
| UT-DeltaRule-111 | Q not 3-D | DeltaRuleTokens | Q zeros([2, 3]) | `Q must be [H, T, Dk], got [2, 3]` | |
| UT-DeltaRule-112 | T = 0 | DeltaRuleTokens | Q, K zeros([2, 0, 4]), V zeros([2, 0, 5]), Beta, G zeros([2, 0]) | `no tokens (T = 0)` | |
| UT-DeltaRule-113 | K wrong shape | DeltaRuleTokens | K zeros([2, 3, 3]) | `K must be the shape of Q [2, 3, 4], got [2, 3, 3]` | |
| UT-DeltaRule-114 | V wrong head count | DeltaRuleTokens | V zeros([1, 3, 5]) | `V must be [2, 3, Dv], got [1, 3, 5]` | |
| UT-DeltaRule-115 | V wrong T | DeltaRuleTokens | V zeros([2, 2, 5]) | `V must be [2, 3, Dv], got [2, 2, 5]` | |
| UT-DeltaRule-116 | Beta wrong shape | DeltaRuleTokens | Beta zeros([2, 4]) | `Beta must be [2, 3], got [2, 4]` | |
| UT-DeltaRule-117 | G wrong shape | DeltaRuleTokens | G zeros([2]) | `G must be [2, 3], got [2]` | |
| UT-DeltaRule-201 | Q wrong kind | DeltaRuleChunked | Q zeros([2, 3, 4], kDouble) | `Q must be kFloat (fp32)` | |
| UT-DeltaRule-202 | K wrong kind | DeltaRuleChunked | K zeros([2, 3, 4], kDouble) | `K must be kFloat (fp32)` | |
| UT-DeltaRule-203 | V wrong kind | DeltaRuleChunked | V zeros([2, 3, 5], kDouble) | `V must be kFloat (fp32)` | |
| UT-DeltaRule-204 | Beta wrong kind | DeltaRuleChunked | Beta zeros([2, 3], kDouble) | `Beta must be kFloat (fp32)` | |
| UT-DeltaRule-205 | G wrong kind | DeltaRuleChunked | G zeros([2, 3], kBFloat16) | `G must be kFloat (fp32)` | |
| UT-DeltaRule-206 | Q not 3-D | DeltaRuleChunked | Q zeros([2, 3]) | `Q must be [H, T, Dk], got [2, 3]` | |
| UT-DeltaRule-207 | T = 0 | DeltaRuleChunked | Q, K zeros([2, 0, 4]), V zeros([2, 0, 5]), Beta, G zeros([2, 0]) | `no tokens (T = 0)` | |
| UT-DeltaRule-208 | K wrong shape | DeltaRuleChunked | K zeros([2, 3, 3]) | `K must be the shape of Q [2, 3, 4], got [2, 3, 3]` | |
| UT-DeltaRule-209 | V wrong T | DeltaRuleChunked | V zeros([2, 2, 5]) | `V must be [2, 3, Dv], got [2, 2, 5]` | |
| UT-DeltaRule-210 | Beta wrong shape | DeltaRuleChunked | Beta zeros([2, 4]) | `Beta must be [2, 3], got [2, 4]` | |
| UT-DeltaRule-211 | G wrong shape | DeltaRuleChunked | G zeros([2]) | `G must be [2, 3], got [2]` | |
| UT-DeltaRule-212 | ChunkSize 0 | DeltaRuleChunked | ChunkSize 0 | `chunk size must be at least 1, got 0` | |
| UT-DeltaRule-213 | Hand case, default chunk | DeltaRuleChunked | T08's hand case: Q Make([1, 1, 1, 2], [1, 2, 2]), K Make([1, 0, 1, 1], [1, 2, 2]), V Make([2, 4, 3, 3], [1, 2, 2]), Beta Make([0.5, 0.5], [1, 2]), G Make([0, 0], [1, 2]); default chunk | exact: O `[1, 2, 2]` `1 2 4 3.5`; State `[1, 2, 2]` `2 1 2.5 0.5` | |
| UT-DeltaRule-214 | Hand case with decay | DeltaRuleChunked | as 213 with G Make([0, Ln(0.5)], [1, 2]) | exact: O `1 2 4.25 4`; State `1.75 1.25 2 1` | |
| UT-DeltaRule-215 | β = 0, zero state | DeltaRuleChunked | `RandomInputs(7, 3, 5, 4, 6, 1.0, …)`, Beta replaced by zeros([3, 5]); ChunkSize 2 | O `[3, 5, 6]`, State `[3, 6, 4]`; `O.abs.max` and `State.abs.max` 0 | |
| UT-DeltaRule-216 | ChunkSize 2, random vs tokens | DeltaRuleChunked | `RandomInputs(20, 2, 5, 4, 6, 1.0, …)` (Dk 4, Dv 6); ChunkSize 2 | O `[2, 5, 6]`, State `[2, 6, 4]`; against tokens(..) ≤ 1e-6 (measured 1.8e-7, 1.2e-7) | |
| UT-DeltaRule-217 | ChunkSize 1, random vs tokens | DeltaRuleChunked | `RandomInputs(16, 4, 50, 16, 16, 1.0, …)`; ChunkSize 1 | against tokens(..) ≤ 1e-6 (measured 1.8e-7) | |
| UT-DeltaRule-218 | ChunkSize 7, padding 6, random vs tokens | DeltaRuleChunked | `RandomInputs(17, 4, 50, 16, 16, 1.0, …)`; ChunkSize 7 (padding 6) | ≤ 1e-6 (measured 1.8e-7) | |
| UT-DeltaRule-219 | T = 1, model sizes vs tokens | DeltaRuleChunked | `RandomInputs(101, 48, 1, 128, 128, 0.1, …)`; default chunk (T = 1) | ≤ 1e-5 (measured 2.4e-7) | |
| UT-DeltaRule-220 | T = 64, no padding, model sizes vs tokens | DeltaRuleChunked | `RandomInputs(164, 48, 64, 128, 128, 0.1, …)`; default (T = 64, no padding) | ≤ 1e-5 (measured 7.2e-7) | |
| UT-DeltaRule-221 | T = 65, padding 63, model sizes vs tokens | DeltaRuleChunked | `RandomInputs(165, 48, 65, 128, 128, 0.1, …)`; default (T = 65, padding 63) | ≤ 1e-5 (measured 5.7e-7) | |
| UT-DeltaRule-222 | Strong decay, model sizes vs tokens | DeltaRuleChunked | `RandomInputs(14, 16, 200, 128, 128, 5.0, …)`; default (strong decay) | ≤ 1e-5 (measured 4.4e-6) | |
| UT-DeltaRule-223 | Default chunk equals ChunkSize 64 | DeltaRuleChunked | `RandomInputs(21, 3, 70, 8, 8, 1.0, …)`; once without ChunkSize, once with 64 | exact: equal O and State | |
| UT-DeltaRule-224 | Gradients of the five inputs vs token form | DeltaRuleChunked | `RandomInputs(22, 4, 70, 16, 16, 1.0, …)`; `ManualSeed(23)`; W := `TTensor.randn([4, 70, 16])`; two copies of the five inputs as new leaves (`detach.Copy.requires_grad_(True)`); backward of `(tokens(..) * W).sum(kNone)` on one and of `(DeltaRuleChunked(.., 16) * W).sum(kNone)` on the other | the five gradients agree ≤ 1e-5 (measured 2.1e-6) | |
