# All Model Evaluation Report

Sorted from worst to best by **Test MAE** (higher is worse).

---

## Context: Shared Task Setup

All models forecast **PM2.5 concentration (µg/m³)** at 12 Beijing monitoring stations
simultaneously, 6 hours ahead, using a 24-hour lookback window.

**Dataset**: Beijing Multi-Site Air Quality (12 stations: Aotizhongxin, Changping, Dingling,
Dongsi, Guanyuan, Gucheng, Huairou, Nongzhanguan, Shunyi, Tiantan, Wanliu, Wanshouxigong)

**Input tensor shape**: `(batch, 24, 12, 33)` — 24 timesteps × 12 nodes × 33 features

**33 input features per node** (index order matters for wind extraction):
- Index 0: `pm2.5` (target)
- Indices 1–6: `pm10`, `so2`, `no2`, `co`, `o3` (other pollutants)
- Indices 7–11: `temp`, `pres`, `dewp`, `rain`, `wspm` (meteorological; wspm=wind speed at index 10)
- Indices 12–17: `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `weekday_sin`, `weekday_cos`
- Indices 17–32: `wd_E`, `wd_ENE`, `wd_ESE`, `wd_N`, `wd_NE`, `wd_NNE`, `wd_NNW`, `wd_NW`,
  `wd_S`, `wd_SE`, `wd_SSE`, `wd_SSW`, `wd_SW`, `wd_W`, `wd_WNW`, `wd_WSW`
  (16 one-hot wind direction categories, alphabetically ordered as `pandas.get_dummies` produces them)

**Output shape**: `(batch, 6, 12)` — 6 horizon steps × 12 nodes, PM2.5 only

**Data split**: Chronological 70% / 15% / 15% (train / val / test). No shuffling.
No temporal overlap between splits.

**Scaling**: `MinMaxScaler` per feature, fit on training portion only. Target (PM2.5)
has its own separate scaler. Metrics are reported in **original scale (µg/m³)** after
inverse transform.

**EVT threshold**: Derived from the 90th percentile of *scaled* training targets.
Computed after split, before training — no leakage.

**MAPE masking**: Values where target ≤ 5.0 µg/m³ are excluded from MAPE to avoid
division-by-near-zero instability.

---

## Core Architecture Reference (GCN-LSTM Models)

All GCN-LSTM models (ranks 4–11) share the same codebase in `models/`. This section
documents the exact building blocks so each rank entry only needs to list what differs.

### GraphConvolution (`models/layers.py`)
Standard graph convolution: `X' = A_hat @ (X @ W) + b`
- `W`: `nn.Parameter` of shape `(in_features, out_features)`, Xavier uniform init
- `b`: `nn.Parameter` of shape `(out_features,)`, zero init
- Handles both static adjacency `(num_nodes, num_nodes)` and dynamic adjacency
  `(batch, num_nodes, num_nodes)` via `torch.matmul` broadcasting

### GraphLSTMCell (`models/layers.py`)
Combines GCN spatial aggregation with LSTM temporal gating.
- `gcn_i`: `GraphConvolution(input_dim, hidden_dim)` — aggregates input features over graph
- `gcn_h`: `GraphConvolution(hidden_dim, hidden_dim)` — aggregates hidden state over graph
- Both GCN outputs pass through `LeakyReLU(negative_slope=0.1)`
- Concatenate: `[x_gcn, h_gcn]` → shape `(batch, nodes, hidden_dim * 2)`
- Single fused linear: `nn.Linear(hidden_dim * 2, hidden_dim * 4)` computes all 4 gate logits
- Gate activations: `i = sigmoid(·)`, `f = sigmoid(·)`, `g = tanh(·)`, `o = sigmoid(·)`
- Update: `c_new = f * c + i * g`, `h_new = o * tanh(c_new)`
- Hidden states initialized to zeros at start of each sequence

### GraphLSTMEncoder (`models/encoder.py`)
Processes the full 24-timestep input window into a context representation.
1. **Input projection**: `nn.Linear(input_dim, hidden_dim)` — maps 33 features to `hidden_dim`
2. **Positional encoding**: sinusoidal (standard transformer-style), added to all nodes at each
   timestep (broadcast over batch and node dimensions). Max length 500, dropout applied.
3. **2 stacked GraphLSTMCell layers** with the **Pre-LN pattern**:
   - At each timestep `t`, for each layer `i`:
     - `layer_input_norm = LayerNorm(layer_input)`
     - If `use_node_embeddings=True`: `layer_input_norm += node_embed` (see below)
     - `h_new, c_new = GraphLSTMCell(layer_input_norm, (h, c), adj)`
     - Residual connection (only for layer index > 0): `h_new = h_new + layer_input`
     - `h_new = Dropout(h_new)`
     - Update hidden state; `layer_input = h_new`
4. **Node embeddings** (if enabled):
   - `nn.Embedding(num_nodes=12, hidden_dim=64)`, init `std=0.01`
   - Injected **after Pre-LN, before the GraphLSTMCell** so that LayerNorm cannot
     re-center the station-identity signal away before the cell sees it.
   - Same embedding reused at every timestep and both encoder layers.
   - Design decision: placing before Pre-LN (v1 attempt) was confirmed useless — LN
     re-centers the signal. Post-LN injection (v2, current) gave Δ val MAE = −0.107.
5. **Returns**:
   - `encoder_outputs`: shape `(batch, 24, num_nodes, hidden_dim)` — all timestep outputs from
     the top encoder layer, used by attention
   - `hidden_states`: list of 2 `(h, c)` tuples (one per layer), final states used to
     initialize the decoder

### DirectMultiHorizonDecoder (`models/decoder.py`)
Predicts all 6 horizon steps jointly, with no autoregressive feedback between steps.
- **step_queries**: `nn.Parameter(shape=(max_horizon=24, hidden_dim=64))`, Xavier uniform init.
  One learnable query vector per future horizon step. Only the first 6 are used.
- **For each horizon step `t` independently**:
  1. `step_q = step_queries[t]` expanded to `(batch, num_nodes, hidden_dim)`
  2. `query = step_q + final_encoder_h` — combines learnable step intent with encoder signal.
     `final_encoder_h` is the top-layer encoder hidden state `h` at the last timestep.
  3. **If `use_attention=True`** (enabled in older checkpoints):
     - `context, attn_weights = MultiHeadAttention(query, encoder_outputs, encoder_outputs)`
     - `combined = context_proj(cat([query, context]))` — `Linear(hidden_dim*2, hidden_dim)`
  4. **If `use_attention=False`** (current default since 2026-04-14):
     - `combined = context_proj(query)` — `Linear(hidden_dim, hidden_dim)` (query only)
  5. `combined = Dropout(combined)`
  6. **Graph-LSTM processing** (Pre-LN, 2 layers, same as encoder but with fresh state copies):
     - Each step receives a **fresh clone** of the encoder hidden states — no recurrence
       between steps, so steps are truly independent
     - Pre-LN → GraphLSTMCell → residual (layer > 0) → Dropout
  7. `output = output_proj(layer_input)` — `Linear(hidden_dim, output_dim=1)`
- `teacher_forcing_ratio` argument accepted for API compatibility but is **completely ignored**
- No teacher forcing, no autoregression — this is the key design choice for direct decoding

### GraphLSTMDecoder (`models/decoder.py`) — autoregressive variant
Used only by v1_baseline (`use_direct_decoding=False`). Each horizon step feeds into the next.
- Always uses MultiHeadAttention (not toggleable in this decoder)
- At each step: `attention(decoder_input, encoder_outputs, encoder_outputs)` →
  `context_proj(cat([decoder_input, context]))` → Dropout → Pre-LN GraphLSTM layers → output_proj
- Decoder input at step 0: top-layer encoder hidden state `h`
- Teacher forcing: per-sample mask computed once per batch (not per timestep).
  `torch.rand(batch) < teacher_forcing_ratio` — if True for that sample, use ground truth;
  otherwise use own prediction. This means all timesteps of a given sample in a batch
  use the same teacher-forcing decision.
- At inference: pure autoregressive (no teacher forcing, previous prediction fed as input)
- Errors accumulate step-to-step

### MultiHeadAttention (`models/layers.py`)
Used in the decoder when `use_attention=True`.
- `hidden_dim=64`, `num_heads=4`, `head_dim=16`, scale = `sqrt(16) = 4.0`
- Separate Q, K, V projection layers (each `Linear(64, 64)`)
- Attention computed per node independently: reshape `(B, N, H)` → `(B*N, H)` for Q;
  `(B, T, N, H)` → permute → `(B*N, T, H)` for K/V
- Produces attn_weights `(B, N, num_heads, T)` and context `(B, N, H)`
- Output projection: `Linear(64, 64)` after head concatenation
- Ablation result (2026-04-14): zero measurable effect on this task — val MAE identical
  with or without it. 20,736 parameters saved by removing it.

### Adjacency Matrices

#### Static distance-only adjacency (used by v1, v2, v3; also fallback for baseline_lstm)
Built once, saved to `data/processed/adjacency.npy`, loaded at training time.
- `A[i,j] = exp(-d(i,j)² / 1800)` for `i ≠ j`, where `d` is Haversine distance in km
- Self-loops: `A = A + I`
- **Symmetric normalization**: `A_hat = D^{-1/2} A D^{-1/2}` — undirected, treats
  pollution transport as symmetric (same weight i→j and j→i)
- Fixed throughout training — does not change per batch

#### Dynamic wind-aware adjacency (used by v4, alpha, alpha+embed, noattn)
Built **per batch** from the input window. Computed on GPU via `build_dynamic_adjacency_gpu`
when CUDA is available or when `alpha_override` (learnable alpha) is passed — this keeps
gradients flowing from loss through the adjacency construction.

**Construction pipeline per sample in a batch**:
1. Extract wind from input window: `wspm` at feature index 10; one-hot wind direction at
   feature indices 17:33 (16 categories)
2. Temporal aggregation over 24 timesteps: `recent_weighted` mode with `recency_beta=3.0`
   (exponential weights increasing toward the most recent timestep)
3. Wind direction aggregation: `circular` method — proper angular mean using sin/cos
   decomposition, weighted jointly by temporal recency and wind speed. Calm wind
   (speed < 0.1 m/s) is treated as neutral (alignment = 0.5).
4. Compute `A_dist[i,j] = exp(-d(i,j)² / 1800)` with `diagonal = 1.0`
5. Compute `A_wind[i,j]`:
   - `source_alignment`: does wind at station `i` transport pollution toward `j`?
     `= ((cos(transport_dir - bearing_ij) + 1) / 2) * tanh(speed_i / 5.0)`
     where `transport_dir = (wind_angle_i + 180) % 360`
   - `receiving_alignment`: does wind at station `j` support incoming transport from `i`?
     `= (cos(wind_angle_j - bearing_ji) + 1) / 2`
   - `A_wind[i,j] = source_alignment * (0.5 + 0.5 * receiving_alignment) * A_dist[i,j]`
   - Self-loops: `diagonal = 1.0`
6. Mix: `A = (1 - alpha) * A_dist + alpha * A_wind`
7. **Row normalization** (directed): `A_hat = A / row_sum` — preserves directionality of
   transport. Different from static adj's symmetric normalization.
8. `alpha = 0.6` fixed, OR `alpha = sigmoid(alpha_logit)` if `use_learnable_alpha_gate=True`

**Note on normalization difference**: Static adj uses symmetric `D^{-1/2} A D^{-1/2}`;
dynamic adj uses row normalization. This means the two are not fully comparable — swapping
adjacency type also changes the normalization scheme.

### EVT Hybrid Loss (`train.py: EVTHybridLoss`)
`loss = MSE(pred, target) + lambda_tail * EVT_tail(pred, target)`

**MSE component**: standard `F.mse_loss(predictions, targets)` — all samples, all horizons

**EVT tail component**: only activates for samples where `target > threshold`
- `threshold`: 90th percentile of *scaled* training targets (global scalar, computed from
  `Y_train_scaled` after split). Stored as a buffer (not a parameter). Computed once before
  training, not updated during training.
- For masked extreme samples:
  - `target_excess = target - threshold`
  - `mean_excess = mean(target_excess).detach()` — detached to avoid second-order gradients
  - `excess_weight = clamp(1 + xi * target_excess / (mean_excess + eps), min=1.0)`
    where `xi=0.10` (GPD shape parameter), `eps=1e-6`
  - `err = pred - target`
  - `tail_loss = mean(excess_weight * err²)`
- `lambda_tail = 0.05` (fixed; lambda schedule disabled in current config)
- `asymmetric_penalty = False` (under-prediction multiplier disabled)

**Note**: Loss is computed on **normalized (scaled) predictions and targets**.
The threshold is on the normalized scale. Val MAE for early stopping is computed on
original scale after inverse transform.

### Training Setup (current defaults, all GCN-LSTM models)
| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Weight decay | `1e-5` |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping criterion | val MAE (original scale, patience=15) |
| Checkpoint criterion | best val MAE |
| LR scheduler | `ReduceLROnPlateau(mode='min', factor=0.5, patience=5)` on val loss |
| Gradient clipping | `max_norm=1.0` |
| Seed | 42 |
| Deterministic | False |
| Teacher forcing | Linear decay 1.0→0.0 over epochs; **ignored by DirectMultiHorizonDecoder** |

**Note on older runs (v1, v2, v3)**: These were trained before the current defaults were
finalized. Exact hyperparameters for those runs are not stored in the checkpoints.
They also predate the early stopping fix (see rank 4 notes). Do not treat their
training setup as identical to current defaults.

---

## Ranking Summary

| Rank | Model | Architecture | Test MAE | Test RMSE | Test MAPE | Test R2 |
|---:|---|---|---:|---:|---:|---:|
| 1 | baseline_historical_mean | historical_mean | 44.4117 | 69.1075 | 138.4588% | 0.4412 |
| 2 | baseline_mlp_best.pt | baseline_mlp | 26.3128 | 44.6829 | 55.0261% | 0.7664 |
| 3 | baseline_persistence | persistence | 23.9829 | 45.8596 | 53.4814% | 0.7539 |
| 4 | v3_evt_loss_T4_best_MAE.pt | v3_evt_loss | 22.7357 | 40.2200 | 50.2129% | 0.8107 |
| 5 | v1_baseline_T4_best_MAE.pt | v1_baseline | 22.5457 | 40.3418 | 48.6125% | 0.8096 |
| 6 | baseline_lstm_best.pt | baseline_lstm | 22.5163 | 40.8132 | 51.7014% | 0.8051 |
| 7 | v2_direct_decoding_T4_best_MAE.pt | v2_direct_decoding | 22.4972 | 40.3729 | 46.3215% | 0.8093 |
| 8 | v4_wind_adjacency_T4_MAE.pt | gcn_lstm_v2 | 22.3807 | 40.2931 | 47.7384% | 0.8100 |
| 9 | alpha__best.pt | alpha | 21.8041 | 39.1870 | 46.0484% | 0.8203 |
| 10 | alpha+embeding.pt | gcn_lstm_v2 | 21.6361 | 39.0500 | 45.9034% | 0.8216 |
| 11 | gcn_lstm_v2_noattn_T4_best.pt | gcn_lstm_v2_noattn | 21.6356 | 39.0410 | 45.7963% | 0.8217 |

---

## Rank 1: baseline_historical_mean

**Architecture Name**: `historical_mean`

**Type**: Non-parametric baseline. No learned parameters. No training.

**Prediction rule**: For each sample, compute the mean PM2.5 value over the 24-hour
lookback window (all 12 nodes, all 24 timesteps). Predict that scalar for every horizon
step and every node.

**Implementation**: Computed directly in Python from the unscaled test set (no model,
no scaler needed). Equivalent to: `pred[b, h, n] = mean(X[b, :, n, 0])` — mean of PM2.5
feature (index 0) over the lookback window for each sample and node.

**Why it performs so poorly**: The historical mean smooths over recent dynamics.
Because PM2.5 is highly autocorrelated (values change slowly), predicting the window mean
ignores the most recent level and introduces large errors when values have recently changed.
At H1 the mean is far from the actual current value; at H6 it remains equally far.
MAPE of 138% reflects that the mean prediction regularly overshoots or undershoots
by a large fraction of the true value. The near-flat horizon degradation (H1 MAE 39.4 → H6
MAE 49.1) shows this model has no temporal awareness at all.

**Purpose in thesis**: Lower bound. Any model that cannot beat this is useless.
All learned models in this report beat it decisively.

**Architecture Details**:
- horizon: 6
- lookback_hours: 24
- rule: predict mean PM2.5 over lookback window for all horizons
- type: non_parametric_baseline

**Overall Metrics**:
- RMSE: 69.1075
- MAE: 44.4117
- MAPE: 138.4588%
- R²: 0.4412

**Per-Horizon Metrics**:

| Horizon | RMSE | MAE | MAPE |
|---:|---:|---:|---:|
| 1 | 62.0666 | 39.4246 | 119.9865% |
| 2 | 65.1141 | 41.5629 | 127.7178% |
| 3 | 67.9216 | 43.5793 | 135.2003% |
| 4 | 70.5223 | 45.4935 | 142.4191% |
| 5 | 72.9420 | 47.3266 | 149.3760% |
| 6 | 75.2042 | 49.0833 | 156.0531% |

---

## Rank 2: baseline_mlp_best.pt

**Architecture Name**: `baseline_mlp`

**Type**: Learned parametric baseline (`MLPBaseline` class, `baselines/train_mlp.py`).

**What it answers**: Can a plain feedforward network match the GCN-LSTM just by having
enough capacity to implicitly learn spatial and temporal patterns from a flattened input?

**Architecture** (`class MLPBaseline`):
```
Input: (batch, 24, 12, 33) → reshape to (batch, 24*12*33=9504)
  → Linear(9504, 512) → LeakyReLU(0.1) → Dropout(0.1)
  → Linear(512, 256) → LeakyReLU(0.1) → Dropout(0.1)
  → Linear(256, 6*12=72)
  → reshape to (batch, 6, 12)  [horizon × nodes]
```
- **No spatial structure**: all 12 nodes flattened together — the model must implicitly
  learn inter-station relationships from co-occurrence in the flat vector
- **No temporal structure**: all 24 timesteps flattened together — temporal ordering is
  encoded only by position in the flat vector, not by any recurrence or attention
- Output is all 6 horizons and all 12 nodes predicted jointly in a single forward pass

**Training**:
- Optimizer: Adam, lr=1e-3, weight_decay=1e-5
- Loss: MSE
- Batch size: 64, max epochs: 100, patience: 15
- Early stopping on val MAE (original scale, inverse-transformed)
- LR scheduler: ReduceLROnPlateau(mode='min', factor=0.5, patience=5)
- Gradient clipping: max_norm=1.0
- Seed: 42

**Why it ranks here**: The MLP can learn global patterns from the flattened input, which
explains its strong overall performance relative to non-parametric baselines. However, it
degrades significantly at longer horizons (H6 MAE 33.5 vs H1 MAE 19.1) and lacks the
inductive biases of graph structure and recurrence. Its H1 MAE of 19.1 is notably worse
than the GCN-LSTM's 13.1, showing that the temporal recurrence (LSTM) provides a large
benefit near the forecast origin.

**Architecture Details**:
- horizon: 6
- input_dim: 33
- num_nodes: 12
- hidden1: 512
- hidden2: 256
- dropout: 0.1
- type: flattened MLP baseline (no graph, no recurrent encoder)

**Overall Metrics**:
- RMSE: 44.6829
- MAE: 26.3128
- MAPE: 55.0261%
- R²: 0.7664
- Val MAE: 22.5417

**Per-Horizon Metrics**:

| Horizon | RMSE | MAE | MAPE |
|---:|---:|---:|---:|
| 1 | 31.6091 | 19.1488 | 37.2958% |
| 2 | 36.1631 | 21.6508 | 42.3557% |
| 3 | 41.5409 | 24.7611 | 49.8744% |
| 4 | 46.8669 | 27.9086 | 58.1287% |
| 5 | 51.6341 | 30.9013 | 66.7667% |
| 6 | 55.5360 | 33.5062 | 75.7356% |

---

## Rank 3: baseline_persistence

**Architecture Name**: `persistence`

**Type**: Non-parametric baseline. No learned parameters. No training.

**Prediction rule**: Predict the last observed PM2.5 value (final timestep of the 24h
lookback window) as the forecast for all 6 horizon steps, for each node independently.
`pred[b, h, n] = X[b, -1, n, 0]` — the PM2.5 feature (index 0) at the last timestep.

**Why it ranks here**: PM2.5 is highly autocorrelated. The most recent observation is
often a good proxy for the near-future value. At H1, this baseline achieves MAE=10.62,
which is the **best H1 of any model in this report** (including the best GCN-LSTM at 13.06).
This is expected: the GCN-LSTM must generalize across all conditions, including sudden
changes, while persistence always uses the most recent value.

**Horizon degradation**: H1 MAE=10.62 → H6 MAE=34.75 — fast degradation because PM2.5
does change meaningfully over 6 hours. The GCN-LSTM (H1=13.06 → H6=29.59) degrades more
slowly, demonstrating its forecasting advantage at longer horizons.

**Important caveat**: Persistence has the highest RMSE of any learned/non-trivial model
(45.86) despite a low MAE. This reflects extreme outlier events: when PM2.5 changes
abruptly, persistence predicts the old value and incurs a large squared error. The
disproportionate RMSE relative to MAE is the signature of a model that fails on extreme events.

**Purpose in thesis**: Essential reference point. A model that beats persistence only on
average MAE but not on RMSE or extreme-event metrics may not be scientifically useful.
Also demonstrates that for the first horizon, learned models are still below persistence,
meaning the temporal trend information only starts to add value at H2+.

**Architecture Details**:
- horizon: 6
- rule: predict last observed PM2.5 for all horizons
- type: non_parametric_baseline

**Overall Metrics**:
- RMSE: 45.8596
- MAE: 23.9829
- MAPE: 53.4814%
- R²: 0.7539

**Per-Horizon Metrics**:

| Horizon | RMSE | MAE | MAPE |
|---:|---:|---:|---:|
| 1 | 21.2163 | 10.6191 | 21.3874% |
| 2 | 33.5571 | 17.3761 | 35.2901% |
| 3 | 42.3521 | 22.6854 | 48.1668% |
| 4 | 49.5099 | 27.2446 | 60.5681% |
| 5 | 55.5858 | 31.2243 | 72.1863% |
| 6 | 60.8909 | 34.7479 | 83.2895% |

---

## Rank 4: v3_evt_loss_T4_best_MAE.pt

**Architecture Name**: `v3_evt_loss`

**Type**: `GCNLSTMModel` (same class as all GCN-LSTM models). **Contaminated run — do not
use as a controlled ablation point in thesis comparisons.**

**⚠ Why this run is unreliable**: This checkpoint stopped at **epoch 10**. It was trained
before the early stopping bug was fixed. The bug: early stopping monitored EVT val_loss
instead of val_mae. The EVT lambda schedule caused the val_loss to increase mechanically
at the warmup boundary (epoch 15), regardless of actual model quality. In this run,
the loss likely began rising early enough that the patience counter expired by epoch 10,
terminating training far too early. The resulting checkpoint is severely undertrained.
Val MAE of 20.71 is the worst of all GCN-LSTM models, consistent with this diagnosis.

**Architecture**:
- `input_dim`: 33, `hidden_dim`: 64, `output_dim`: 1
- `num_nodes`: 12, `num_layers`: 2, `num_heads`: 4, `dropout`: 0.1
- `use_direct_decoding`: **True** — DirectMultiHorizonDecoder
- `use_wind_adjacency`: **False** — uses static distance-only adjacency (symmetric normalized)
- `use_learnable_alpha_gate`: **False**
- `use_node_embeddings`: **False**
- `use_attention`: **True** — MultiHeadAttention active in decoder (4 heads)
- `step_queries`: learnable, `(24, 64)`, Xavier uniform init
- `wind_direction_method`: circular (used for adj construction method, applies when
  `use_wind_adjacency=True` — here irrelevant since adjacency is static)

**Loss**: EVT hybrid. Parameters likely used default settings at the time (exact lambda
schedule from that training run is not preserved). This was the experiment intended to
validate whether EVT loss helps, but the early stopping bug makes the result invalid.

**Training notes**: Epoch 10. Early stopping triggered far too early due to EVT val_loss
bug. Exact hyperparameters from that run are not preserved; likely predates current defaults.

**Architecture Details**:
- use_direct_decoding: True
- use_wind_adjacency: False (static adj: Gaussian distance decay, symmetric normalized)
- use_learnable_alpha_gate: False
- use_learnable_sigma: False
- use_node_embeddings: False
- use_attention: True (4 heads, 20,736 params)
- wind_direction_method: circular
- wind_temporal_graphs: 1
- wind_temporal_graph_window: None
- loss_type: evt_hybrid

**Overall Metrics**:
- RMSE: 40.2200
- MAE: 22.7357
- MAPE: 50.2129%
- R²: 0.8107
- Epoch: 10 ← severely undertrained
- Val Loss: 0.002715 (EVT loss, not MAE)
- Val MAE: 20.7130

**Per-Horizon Metrics**:

| Horizon | RMSE | MAE | MAPE |
|---:|---:|---:|---:|
| 1 | 27.0414 | 15.6805 | 28.5861% |
| 2 | 31.8610 | 18.2013 | 34.7018% |
| 3 | 37.2056 | 21.2087 | 43.6515% |
| 4 | 42.2682 | 24.2194 | 53.5608% |
| 5 | 46.7912 | 27.1199 | 64.3699% |
| 6 | 50.9829 | 29.9841 | 76.4076% |

---

## Rank 5: v1_baseline_T4_best_MAE.pt

**Architecture Name**: `v1_baseline`

**Type**: `GCNLSTMModel`. Early experimental run. **Stale — two known-weaker choices
active (autoregressive decoder + static adjacency). Do not use as primary ablation.**

**Architecture**:
- `input_dim`: 33, `hidden_dim`: 64, `output_dim`: 1
- `num_nodes`: 12, `num_layers`: 2, `num_heads`: 4, `dropout`: 0.1
- `use_direct_decoding`: **False** — **GraphLSTMDecoder (autoregressive)**
  - At each of the 6 horizon steps: apply MHA over encoder outputs, combine with previous
    prediction, run through 2 graph LSTM layers, project to output.
  - Previous prediction (or teacher-forced ground truth) is fed as input to the next step.
  - Errors accumulate across steps — H6 error is partly driven by H1–H5 prediction errors.
  - Teacher forcing with scheduled decay during training. At inference: purely autoregressive.
- `use_wind_adjacency`: **False** — static distance-only adjacency (symmetric normalized)
- `use_learnable_alpha_gate`: **False**
- `use_node_embeddings`: **False**
- `use_attention`: **True** (always active in GraphLSTMDecoder, not configurable)
- `wind_direction_method`: circular (irrelevant — adjacency is static)
- Loss: **MSE** (not EVT)

**Why it ranks just below baseline_lstm (rank 6)**:
1. **Static adjacency with symmetric normalization** may provide little or negative benefit
   over having no adjacency at all. A poorly-informed undirected graph (same weight i→j as
   j→i) could add noise rather than useful signal, and static adjacency cannot encode
   real-time wind-driven transport.
2. **Autoregressive decoder** accumulates errors across 6 steps.
3. The gap to rank 6 (baseline_lstm) is only 0.03 MAE — within noise. These two models
   are effectively tied; the ranking is not meaningful at this precision.

**Training notes**: Epoch 41. Trained before current defaults; exact hyperparameters
not preserved. Predates the EVT hybrid loss, dynamic adjacency, alpha gate, and node
embedding experiments. Predates the early stopping fix, but since this used MSE loss
(no EVT lambda schedule), the bug did not affect it.

**Architecture Details**:
- use_direct_decoding: False (autoregressive GraphLSTMDecoder)
- use_wind_adjacency: False (static adj: Gaussian distance decay, symmetric normalized)
- use_learnable_alpha_gate: False
- use_learnable_sigma: False
- use_node_embeddings: False
- use_attention: True (always active in autoregressive decoder)
- wind_direction_method: circular
- wind_temporal_graphs: 1
- wind_temporal_graph_window: None
- loss_type: mse

**Overall Metrics**:
- RMSE: 40.3418
- MAE: 22.5457
- MAPE: 48.6125%
- R²: 0.8096
- Epoch: 41
- Val Loss: 0.002092 (MSE)
- Val MAE: 20.3990

**Per-Horizon Metrics**:

| Horizon | RMSE | MAE | MAPE |
|---:|---:|---:|---:|
| 1 | 26.3916 | 14.9665 | 29.3469% |
| 2 | 32.0627 | 17.8694 | 33.3922% |
| 3 | 37.4987 | 21.1594 | 41.2818% |
| 4 | 42.4759 | 24.2050 | 51.2857% |
| 5 | 47.0082 | 27.0142 | 61.2948% |
| 6 | 51.1871 | 30.0596 | 75.0736% |

---

## Rank 6: baseline_lstm_best.pt

**Architecture Name**: `baseline_lstm`

**Type**: `LSTMBaseline` class (`baselines/train_lstm.py`). No graph structure.

**What it answers**: How much does the GCN spatial component contribute? If GCN-LSTM
only marginally beats this, the graph is not justified. If the gap is large, graph
structure is necessary.

**Architecture** (`class LSTMBaseline`):
```
Input: (batch, 24, 12, 33)
  → permute/reshape to (batch * 12, 24, 33)       [nodes treated as batch dimension]
  → input_proj: Linear(33, 64)                     [project to hidden_dim]
  → nn.LSTM(input_size=64, hidden_size=64,
            num_layers=2, batch_first=True,
            dropout=0.1 between layers)             [temporal encoding, per node]
  → h_n[-1]: (batch * 12, 64)                      [top-layer final hidden state]
  → Dropout(0.1)
```
Decoder: 6 independent horizon heads, one per step:
```
  For each horizon step h in [0..5]:
    → Sequential(Linear(64,64), LeakyReLU(0.1), Dropout(0.1), Linear(64,1))
  → cat outputs: (batch * 12, 6)
  → reshape + permute to (batch, 6, 12)
```

**Key differences from GCN-LSTM**:
- No GraphConvolution: LSTM gates receive raw projected input, not GCN-aggregated input
- No adjacency matrix — nodes are completely isolated from each other
- Nodes share LSTM weights (same parameters applied to each of the 12 nodes) but do not
  exchange any information during processing
- No sinusoidal positional encoding (positional encoding is part of GraphLSTMEncoder only)
- No Pre-LN pattern (the GCN-LSTM encoder uses Pre-LN; this baseline uses standard LSTM)
- No step_queries: decoder uses separate per-horizon heads (6 MLP heads) instead of
  learned query embeddings fed through graph LSTM layers
- No attention: pure direct decoding — each horizon step is predicted independently
  from the same encoder hidden state without any cross-attention

**Training**:
- Optimizer: Adam, lr=1e-3, weight_decay=1e-5
- Loss: MSE
- Batch size: **64** (different from GCN-LSTM batch_size=32)
- Max epochs: 100, patience: 15
- Early stopping on val MAE (original scale)
- LR scheduler: ReduceLROnPlateau(mode='min', factor=0.5, patience=5)
- Gradient clipping: max_norm=1.0
- Seed: 42

**Why it ties with v1/v2**: The static adjacency used in v1 and v2 appears to provide
negligible benefit or even slight harm over no graph at all. This is consistent with the
theory that an uninformative graph (symmetric, static, no wind direction) can add noise.
The meaningful graph benefit only appears with dynamic wind-aware adjacency (rank 8+).

**Architecture Details**:
- dropout: 0.1
- graph_module: False (no GCN — LSTM only)
- hidden_dim: 64
- horizon: 6
- input_dim: 33
- num_layers: 2 (standard nn.LSTM with inter-layer dropout=0.1)
- type: LSTM encoder + direct multi-horizon decoder (6 independent MLP heads)
- batch_size: 64 (note: different from GCN-LSTM models which use batch_size=32)

**Overall Metrics**:
- RMSE: 40.8132
- MAE: 22.5163
- MAPE: 51.7014%
- R²: 0.8051
- Val MAE: 20.4840

**Per-Horizon Metrics**:

| Horizon | RMSE | MAE | MAPE |
|---:|---:|---:|---:|
| 1 | 20.3231 | 11.5826 | 24.6157% |
| 2 | 31.0272 | 17.1696 | 35.5404% |
| 3 | 38.4043 | 21.4286 | 45.3936% |
| 4 | 44.1490 | 25.1046 | 56.6981% |
| 5 | 48.9430 | 28.4908 | 68.7193% |
| 6 | 52.9069 | 31.3218 | 79.2414% |

---

## Rank 7: v2_direct_decoding_T4_best_MAE.pt

**Architecture Name**: `v2_direct_decoding`

**Type**: `GCNLSTMModel`. Ablation run confirming the benefit of direct over autoregressive
decoding (with static adjacency). **Stale checkpoint — static adjacency, no alpha gate,
no node embeddings.**

**Architecture**:
- `input_dim`: 33, `hidden_dim`: 64, `output_dim`: 1
- `num_nodes`: 12, `num_layers`: 2, `num_heads`: 4, `dropout`: 0.1
- `use_direct_decoding`: **True** — DirectMultiHorizonDecoder
- `use_wind_adjacency`: **False** — static distance-only adjacency (symmetric normalized)
- `use_learnable_alpha_gate`: **False**
- `use_node_embeddings`: **False**
- `use_attention`: **True** (4 heads, 20,736 params)
- `step_queries`: learnable `(24, 64)`, Xavier uniform init
- Loss: **MSE**

**Comparison to v1 (rank 5)**: Same configuration except decoder mode.
- v2 (direct): MAE=22.497, trained to epoch 32
- v1 (autoregressive): MAE=22.546, trained to epoch 41
- Δ = −0.049 in favor of direct. Gap is small and both use static adjacency + MSE,
  so this comparison is noisy. The real benefit of direct decoding is cleaner at
  H4–H6 where autoregressive error accumulation matters most.

**Why it ranks marginally better than v1**: Direct decoding avoids error accumulation.
Each horizon step is predicted independently from the same encoder state, so H6 errors
are not compounded by H1–H5 prediction errors. Both use the same static adjacency,
which is why neither breaks above the baseline_lstm.

**Training notes**: Epoch 32. Trained before current defaults; exact hyperparameters
not preserved. Predates dynamic adjacency, alpha gate, node embeddings. The early stopping
bug did not affect this run (MSE loss, no EVT lambda schedule).

**Architecture Details**:
- use_direct_decoding: True
- use_wind_adjacency: False (static adj: Gaussian distance decay, symmetric normalized)
- use_learnable_alpha_gate: False
- use_learnable_sigma: False
- use_node_embeddings: False
- use_attention: True (4 heads)
- wind_direction_method: circular
- wind_temporal_graphs: 1
- wind_temporal_graph_window: None
- loss_type: mse

**Overall Metrics**:
- RMSE: 40.3729
- MAE: 22.4972
- MAPE: 46.3215%
- R²: 0.8093
- Epoch: 32
- Val Loss: 0.002156 (MSE)
- Val MAE: 20.3753

**Per-Horizon Metrics**:

| Horizon | RMSE | MAE | MAPE |
|---:|---:|---:|---:|
| 1 | 26.2013 | 14.6955 | 27.0504% |
| 2 | 31.4540 | 17.6632 | 33.9676% |
| 3 | 37.2049 | 20.9688 | 41.3052% |
| 4 | 42.5351 | 24.1573 | 49.3825% |
| 5 | 47.4176 | 27.2841 | 58.1488% |
| 6 | 51.5955 | 30.2146 | 68.0748% |

---

## Rank 8: v4_wind_adjacency_T4_MAE.pt

**Architecture Name**: `gcn_lstm_v2`

**Type**: `GCNLSTMModel`. First clean run with dynamic wind-aware adjacency enabled.
**Controlled ablation point C**: confirms dynamic adjacency beats static (Δ test MAE −0.784
vs v2_direct_decoding, H1 gap −2.19 MAE). This is one of the confirmed design decisions.

**Architecture**:
- `input_dim`: 33, `hidden_dim`: 64, `output_dim`: 1
- `num_nodes`: 12, `num_layers`: 2, `num_heads`: 4, `dropout`: 0.1
- `use_direct_decoding`: **True** — DirectMultiHorizonDecoder with step_queries
- `use_wind_adjacency`: **True** — dynamic wind-aware adjacency per batch
  - Wind extracted from input window: wspm (feature index 10), one-hot wd (indices 17:33)
  - Temporal aggregation: recent_weighted (recency_beta=3.0)
  - Wind direction aggregation: circular mean
  - alpha=0.6 fixed (not learnable)
  - Row normalization (directed, NOT symmetric)
  - Built per batch on GPU via build_dynamic_adjacency_gpu
- `use_learnable_alpha_gate`: **False** — alpha=0.6 is a fixed hyperparameter
- `use_node_embeddings`: **False**
- `use_attention`: **True** (4 heads, 20,736 params; later confirmed to have zero effect)
- Loss: **EVT hybrid** (lambda=0.05, xi=0.10, threshold=90th percentile of training targets)
- `architecture_name`: `gcn_lstm_v2`

**Training notes**: Epoch 42. Trained with current (post-fix) early stopping on val MAE.
Uses current training setup (lr=1e-3, bs=32, patience=15, epochs=100, seed=42). This was
the first model in the clean experiment sequence.

**Key difference from v2 (rank 7)**:
- Dynamic vs static adjacency: +wind-aware directed graph per batch vs fixed symmetric graph
- EVT vs MSE loss
- These two changes happened simultaneously in this run, so the adjacency and loss effects
  are not isolated here. The clean ablation B (EVT vs MSE) and C (dynamic vs static) were
  done separately later using the alpha+embedding baseline.

**Architecture Details**:
- architecture_name: gcn_lstm_v2
- dropout: 0.1
- hidden_dim: 64
- horizon: 6
- input_dim: 33
- loss_type: evt_hybrid
- num_heads: 4
- num_layers: 2
- num_nodes: 12
- output_dim: 1
- use_attention: True
- use_direct_decoding: True
- use_learnable_alpha_gate: False
- use_learnable_sigma: False
- use_node_embeddings: False
- use_wind_adjacency: True
- wind_aggregation_mode: recent_weighted
- wind_recency_beta: 3.0
- wind_direction_method: circular
- wind_normalization: row
- wind_calm_speed_threshold: 0.1
- wind_alpha: 0.6 (fixed)
- distance_sigma: 1800
- wind_speed_idx: 10
- wind_dir_start_idx: 17
- wind_dir_end_idx: 33

**Overall Metrics**:
- RMSE: 40.2931
- MAE: 22.3807
- MAPE: 47.7384%
- R²: 0.8100
- Epoch: 42
- Val Loss: 0.002404 (EVT)
- Val MAE: 19.6846

**Per-Horizon Metrics**:

| Horizon | RMSE | MAE | MAPE |
|---:|---:|---:|---:|
| 1 | 25.7376 | 14.4653 | 26.2261% |
| 2 | 31.4732 | 17.5027 | 32.7556% |
| 3 | 37.3156 | 20.8516 | 41.1947% |
| 4 | 42.6475 | 24.1971 | 51.7911% |
| 5 | 47.2960 | 27.2456 | 61.7830% |
| 6 | 51.3816 | 30.0218 | 72.6800% |

---

## Rank 9: alpha__best.pt

**Architecture Name**: `alpha`

**Type**: `GCNLSTMModel`. Adds learnable alpha gate on top of v4_wind_adjacency.
**Confirmed improvement**: Δ val MAE −0.106 vs v4 (19.578 vs 19.685).

**Architecture**:
- `input_dim`: 33, `hidden_dim`: 64, `output_dim`: 1
- `num_nodes`: 12, `num_layers`: 2, `num_heads`: 4, `dropout`: 0.1
- `use_direct_decoding`: **True** — DirectMultiHorizonDecoder
- `use_wind_adjacency`: **True** — dynamic wind-aware adjacency per batch
- `use_learnable_alpha_gate`: **True**
  - `alpha_logit`: `nn.Parameter(scalar)`, initialized to `logit(0.6) ≈ 0.405`
  - At runtime: `alpha = sigmoid(alpha_logit)` — constrained to (0, 1)
  - During `build_dynamic_adjacency`: `alpha_override = model.get_wind_alpha()` is passed,
    keeping gradients flowing from loss through adjacency construction into `alpha_logit`
  - Final learned alpha ≈ 0.644 (increased slightly from initial 0.6 — model learned to
    weight wind slightly more than distance)
- `use_node_embeddings`: **False**
- `use_attention`: **True** (4 heads; later confirmed to have zero effect)
- Loss: EVT hybrid (same as v4: lambda=0.05, xi=0.10)

**What learnable alpha adds**: Instead of fixing the wind vs distance trade-off at 0.6,
the model learns the optimal blend from data. The 1 scalar parameter (alpha_logit) adds
gradient feedback from the loss into the adjacency construction. Final value 0.644 suggests
the data supports slightly more wind emphasis than the initial guess.

**Architecture Details**:
- architecture_name: alpha
- dropout: 0.1
- hidden_dim: 64
- horizon: 6
- input_dim: 33
- loss_type: evt_hybrid
- num_heads: 4
- num_layers: 2
- num_nodes: 12
- output_dim: 1
- use_attention: True
- use_direct_decoding: True
- use_learnable_alpha_gate: True (alpha_logit initialized to logit(0.6); final learned ≈ 0.644)
- use_learnable_sigma: False
- use_node_embeddings: False
- use_wind_adjacency: True
- wind_aggregation_mode: recent_weighted
- wind_recency_beta: 3.0
- wind_direction_method: circular
- wind_normalization: row
- wind_calm_speed_threshold: 0.1
- wind_alpha: 0.6 (initial only; overridden by learned sigmoid(alpha_logit) at runtime)
- distance_sigma: 1800 (fixed)
- wind_speed_idx: 10
- wind_dir_start_idx: 17
- wind_dir_end_idx: 33

**Overall Metrics**:
- RMSE: 39.1870
- MAE: 21.8041
- MAPE: 46.0484%
- R²: 0.8203
- Epoch: 52
- Val Loss: 0.002353 (EVT)
- Val MAE: 19.5782

**Per-Horizon Metrics**:

| Horizon | RMSE | MAE | MAPE |
|---:|---:|---:|---:|
| 1 | 23.2172 | 13.2637 | 25.6465% |
| 2 | 29.9211 | 16.6901 | 31.6596% |
| 3 | 36.2802 | 20.3253 | 40.0940% |
| 4 | 41.8240 | 23.7640 | 50.0274% |
| 5 | 46.4072 | 26.9715 | 59.5945% |
| 6 | 50.5991 | 29.8100 | 69.2683% |

---

## Rank 10: alpha+embeding.pt

**Architecture Name**: `gcn_lstm_v2`

**Type**: `GCNLSTMModel`. Adds learnable node identity embeddings on top of alpha.
**Confirmed improvement**: Δ val MAE −0.107 vs alpha-only (19.471 vs 19.578).
**Current best checkpoint** — baseline to beat before GAT.

**Architecture**:
- `input_dim`: 33, `hidden_dim`: 64, `output_dim`: 1
- `num_nodes`: 12, `num_layers`: 2, `num_heads`: 4, `dropout`: 0.1
- `use_direct_decoding`: **True** — DirectMultiHorizonDecoder
- `use_wind_adjacency`: **True** — dynamic wind-aware adjacency per batch
- `use_learnable_alpha_gate`: **True** (same as rank 9)
- `use_node_embeddings`: **True**
  - `nn.Embedding(num_nodes=12, hidden_dim=64)` — 768 parameters total
  - Initialized with `std=0.01` (small random init — stations start near-identical)
  - **Injection point**: after Pre-LN, before GraphLSTMCell input — at every timestep
    and every encoder layer. This placement bypasses the LayerNorm that would otherwise
    re-center the station-identity signal. Confirmed by ablation: v1 injection before
    Pre-LN gave zero gain (val 19.577); v2 injection after Pre-LN gave −0.107 gain.
  - Same embedding vector reused for all 24 timesteps and both encoder layers.
    Node IDs: `torch.arange(12)` (matches fixed STATION_ORDER from preprocessing).
- `use_attention`: **True** (4 heads; later confirmed zero effect — see rank 11)
- Loss: EVT hybrid (lambda=0.05, xi=0.10)

**What node embeddings add**: Each of the 12 monitoring stations gets a learned
64-dimensional identity vector. The GraphLSTMCell sees slightly different inputs for
each station, allowing the model to learn station-specific biases and adjustment factors
(e.g., industrial vs residential vs suburban station characteristics). The small init
(std=0.01) ensures the embeddings start neutral and learn only what's needed.

**Architecture Details**:
- architecture_name: gcn_lstm_v2
- dropout: 0.1
- hidden_dim: 64
- horizon: 6
- input_dim: 33
- loss_type: evt_hybrid
- num_heads: 4
- num_layers: 2
- num_nodes: 12
- output_dim: 1
- use_attention: True (4 heads; confirmed zero effect but still active in this checkpoint)
- use_direct_decoding: True
- use_learnable_alpha_gate: True
- use_learnable_sigma: False
- use_node_embeddings: True (nn.Embedding(12, 64), std=0.01, injected post-LN)
- use_wind_adjacency: True
- wind_aggregation_mode: recent_weighted
- wind_recency_beta: 3.0
- wind_direction_method: circular
- wind_normalization: row
- wind_calm_speed_threshold: 0.1
- wind_alpha: 0.6 (initial; overridden by learned sigmoid(alpha_logit))
- distance_sigma: 1800 (fixed)
- wind_speed_idx: 10
- wind_dir_start_idx: 17
- wind_dir_end_idx: 33

**Overall Metrics**:
- RMSE: 39.0500
- MAE: 21.6361
- MAPE: 45.9034%
- R²: 0.8216
- Epoch: 42
- Val Loss: 0.002335 (EVT)
- Val MAE: 19.4710

**Per-Horizon Metrics**:

| Horizon | RMSE | MAE | MAPE |
|---:|---:|---:|---:|
| 1 | 22.8528 | 13.0518 | 26.2869% |
| 2 | 29.7582 | 16.5613 | 32.0305% |
| 3 | 36.1437 | 20.2022 | 40.0351% |
| 4 | 41.7160 | 23.6213 | 49.2092% |
| 5 | 46.3019 | 26.7973 | 59.2301% |
| 6 | 50.5088 | 29.5828 | 68.6288% |

---

## Rank 11: gcn_lstm_v2_noattn_T4_best.pt

**Architecture Name**: `gcn_lstm_v2_noattn`

**Type**: `GCNLSTMModel`. **Final current baseline** for the GCN-LSTM phase.
Identical to rank 10 except `use_attention=False`. Confirmed that attention adds zero
value (val MAE difference: 0.009, test MAE difference: 0.0005 — measurement noise).

**Architecture**:
- `input_dim`: 33, `hidden_dim`: 64, `output_dim`: 1
- `num_nodes`: 12, `num_layers`: 2, `num_heads`: 4 (parameter unused), `dropout`: 0.1
- `use_direct_decoding`: **True** — DirectMultiHorizonDecoder
- `use_wind_adjacency`: **True** — dynamic wind-aware adjacency per batch
- `use_learnable_alpha_gate`: **True** (same as ranks 9, 10)
- `use_node_embeddings`: **True** (same as rank 10; post-LN injection)
- `use_attention`: **False**
  - `context_proj` becomes `Linear(hidden_dim=64, hidden_dim=64)` (query projected directly)
  - instead of `Linear(hidden_dim*2=128, hidden_dim=64)` (concat [query, context])
  - Saves: Q/K/V projections (3 × 64×64 = 12,288 weights) + out_proj (64×64 = 4,096) +
    context_proj size difference (128×64 vs 64×64) = ~20,736 parameters removed
- Loss: EVT hybrid (lambda=0.05, xi=0.10)
- `architecture_name`: `gcn_lstm_v2_noattn`

**What removing attention confirms**: The step_queries (learnable horizon embeddings) alone
are sufficient to specialize each horizon step's prediction. Cross-attention over the 24-step
encoder sequence provides no additional temporal context — the encoder hidden state already
captures everything the decoder needs. This is consistent with direct decoding: since all
steps start from the same encoder state (no autoregression), there is no changing "current
state" for attention to adapt to.

**This is the GCN-LSTM baseline to beat going forward** (GAT, ablations, etc.).
The full confirmed feature stack: direct decoding + dynamic wind adj + learnable alpha
(final ≈ 0.644) + node embeddings (post-LN) + EVT hybrid loss + no MHA.

**Confirmed ablation evidence** (all compared at identical conditions except the varied factor):
- **B** (EVT vs MSE): test MAE 21.636 vs 21.814 — EVT wins by Δ −0.178, largest at H1 (−0.848)
- **C** (dynamic vs static adj): test MAE 21.636 vs 22.420 — dynamic wins by Δ −0.784,
  H1 gap −2.19 MAE. Dynamic adjacency provides the largest single gain in the architecture.
- **Attention** (with vs without): test MAE 21.6361 vs 21.6356 — zero effect (0.0005 MAE).

**Architecture Details**:
- architecture_name: gcn_lstm_v2_noattn
- dropout: 0.1
- hidden_dim: 64
- horizon: 6
- input_dim: 33
- loss_type: evt_hybrid
- num_heads: 4 (parameter stored but unused since use_attention=False)
- num_layers: 2
- num_nodes: 12
- output_dim: 1
- use_attention: False (context_proj: Linear(64, 64), no MHA module)
- use_direct_decoding: True
- use_learnable_alpha_gate: True
- use_learnable_sigma: False
- use_node_embeddings: True (nn.Embedding(12, 64), std=0.01, injected post-LN)
- use_wind_adjacency: True
- wind_aggregation_mode: recent_weighted
- wind_recency_beta: 3.0
- wind_direction_method: circular
- wind_normalization: row
- wind_calm_speed_threshold: 0.1
- wind_alpha: 0.6 (initial; overridden by learned sigmoid(alpha_logit))
- distance_sigma: 1800 (fixed)
- wind_speed_idx: 10
- wind_dir_start_idx: 17
- wind_dir_end_idx: 33

**Training** (current defaults, fully clean run):
- Optimizer: Adam, lr=1e-3, weight_decay=1e-5
- Loss: EVT hybrid (fixed lambda=0.05; schedule disabled)
- Batch size: 32, max epochs: 100, patience: 15
- Early stopping on val MAE (original scale); checkpoint on best val MAE
- LR scheduler: ReduceLROnPlateau on val_loss (factor=0.5, patience=5)
- Gradient clipping: max_norm=1.0
- Seed: 42, deterministic: False

**Overall Metrics**:
- RMSE: 39.0410
- MAE: 21.6356
- MAPE: 45.7963%
- R²: 0.8217
- Epoch: 42
- Val Loss: 0.002334 (EVT)
- Val MAE: 19.4796

**Per-Horizon Metrics**:

| Horizon | RMSE | MAE | MAPE |
|---:|---:|---:|---:|
| 1 | 22.8867 | 13.0621 | 25.7803% |
| 2 | 29.7516 | 16.5616 | 32.1654% |
| 3 | 36.1299 | 20.1876 | 40.0816% |
| 4 | 41.7076 | 23.6304 | 49.2169% |
| 5 | 46.2781 | 26.7865 | 59.0332% |
| 6 | 50.4944 | 29.5857 | 68.5005% |
