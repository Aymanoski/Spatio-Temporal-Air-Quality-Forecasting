# Experiment Log: Transformer Phase
## Spatio-Temporal PM2.5 Forecasting — Bachelor's Thesis

**Scope:** All experiments conducted after switching from the GCN-LSTM backbone to a Transformer-based architecture. Covers architecture search, normalization, feature engineering, training optimization, calibration, and ordering experiments.

**Dataset:** Beijing Multi-Site Air Quality, 12 monitoring stations, PM2.5 target.  
**Task:** Forecast PM2.5 at all 12 stations, 6 hours ahead, from a 24-hour lookback window.  
**Input shape:** `(batch, 24, 12, 33)` → **Output shape:** `(batch, 6, 12)`  
**Primary metric:** Test MAE in original µg/m³ scale (after log1p inverse transform).  
**Noise floor:** Approximately 0.3–0.5 MAE at single seed. Differences below this threshold are treated as statistical ties.

---

## Baseline Reference (GCN-LSTM Final)

Before the Transformer phase, the best result was the GCN-LSTM model with learnable alpha gate, node embeddings, dynamic wind-aware adjacency, EVT hybrid loss, and no decoder multi-head attention.

| Metric | Value |
|---|---:|
| Test MAE | 21.636 |
| Test RMSE | 39.050 |
| R² | 0.822 |

This serves as the starting point for the entire Transformer phase.

---

## Phase 1 — Architecture Switch: Temporal Backbone and Spatial Module

### Session 2026-04-15

---

#### Exp T-01 — GraphTransformer + GCN

**What changed:** The GCN-LSTM recurrent backbone was replaced with a compact Transformer encoder. The spatial module remained GCN (same as before).

**Why it was tried:** The GCN-LSTM was slow (~95s/epoch) and LSTM-based recurrence is known to be inefficient compared to Transformer attention. The question was whether accuracy would be preserved at significantly lower training cost.

**What it does:** Instead of processing node features through an LSTM cell at each timestep, the model first applies a shared GCN layer to each of the 24 input timesteps (vectorised as one batched matrix multiply), then reshapes to a per-node sequence `(B×N, T, H)`, applies sinusoidal positional encoding, and passes through a 2-layer Transformer encoder with 4 heads and Pre-LN. The last token of the Transformer sequence serves as the node summary, which is then decoded by a direct multi-horizon head.

**Result:**
| Metric | GCN-LSTM | GraphTransformer+GCN |
|---|---:|---:|
| Test MAE | 21.636 | 21.686 |
| Test RMSE | 39.050 | 39.077 |
| Training time | ~95s/epoch | ~11s/epoch |

**Observations:** Statistical tie on all accuracy metrics. The Transformer is 8–9× faster. The temporal backbone is not the performance bottleneck. This confirmed the switch was safe on efficiency grounds without sacrificing accuracy.

---

#### Exp T-02 — GraphTransformer + GAT (v1)

**What changed:** The GCN spatial layer was replaced with a Graph Attention Network (GATv1). The temporal backbone remained the same Transformer.

**Why it was tried:** GCN uses a fixed normalized adjacency as the spatial mixing weight. GAT learns attention weights per edge, meaning it can selectively emphasize which neighboring stations are most relevant for each node. With dynamic wind-aware adjacency already providing directional structure, GAT was expected to better exploit that signal by learning attention on top of it.

**What it does:** For each node, GAT computes a scalar attention score toward each neighbor using a decomposed formulation: `e_ij = LeakyReLU(a_src^T Wh_i + a_tgt^T Wh_j)`. The wind-aware adjacency matrix is used as an additive bias on these scores — positive adjacency values add signal, zero adjacency applies a −∞ mask (hard block). Multi-head attention is used with 4 heads, and outputs are concatenated back to the original hidden dimension.

**Result:**
| Metric | GT + GCN | GT + GAT v1 |
|---|---:|---:|
| Test MAE | 21.686 | **21.184** |
| Test RMSE | 39.077 | **38.074** |
| R² | 0.821 | 0.830 |

**Observations:** Clear improvement: Δ MAE = −0.502, Δ RMSE = −1.003. GAT's learned attention meaningfully outperforms fixed normalized aggregation given the wind-aware adjacency structure. This became the new best model and the canonical architecture for all subsequent experiments.

**Key thesis finding:** The spatial module, not the temporal backbone, is the performance bottleneck. Replacing GCN with GAT produced the largest single architectural gain in the Transformer phase.

---

### Session 2026-04-16 (s1) — Spatial Architecture Variants

---

#### Exp T-03 — 2-Layer GAT

**What changed:** `num_gat_layers` increased from 1 to 2.

**Why it was tried:** Two spatial message-passing steps could allow information to propagate across two hops (e.g., station A → station B → station C), potentially capturing longer-range transport chains.

**What it does:** Applies two sequential GAT layers before feeding into the Transformer, each with its own attention weights.

**Result:** Val MAE 19.413, Test MAE **21.526**, RMSE 38.547 — worse.

**Observations:** Over-smoothing on the 12-node fully-connected graph. With only 12 nodes and wind-aware adjacency already providing strong directional priors, a second message-passing step averages away informative spatial structure. Single-layer GAT is optimal for this graph size.

---

#### Exp T-04 — GATv2 (Dynamic Attention)

**What changed:** Switched from GATv1 (static attention formulation) to GATv2. In GATv2, attention is computed as `a^T LeakyReLU(W_src h_i + W_dst h_j)` — a dynamic formulation that mixes source and target features before scoring, as opposed to GATv1's additive decomposition.

**Why it was tried:** GATv2 was shown in the original paper to be strictly more expressive than GATv1. The hypothesis was that dynamic attention might capture more complex spatial interactions.

**Result:** Val MAE 19.505, Test MAE **21.170**, RMSE ≈ — statistical tie with GATv1 (21.184).

**Observations:** The wind-aware adjacency bias dominates the attention computation. Whether the attention formulation is static or dynamic does not matter much when the adjacency already provides strong directional priors. GATv1 retained as default (simpler, same accuracy).

---

#### Exp T-05 — Iso-Capacity GAT (hidden_dim=96)

**What changed:** Hidden dimension increased from 64 to 96 while keeping the same architecture, to match parameter count more closely with the GCN-LSTM.

**Why it was tried:** To test whether the GT+GAT result was capacity-limited rather than architecture-limited.

**Result:** Val MAE 19.115, Test MAE **21.142**, RMSE 38.458. Val/test gap widened (2.03 vs 1.71). Approximately 2× parameter count.

**Observations:** Marginal MAE gain but RMSE worse, wider generalization gap. The model at h=64 already has sufficient capacity for this 12-node, 33-feature dataset. Additional parameters increase overfitting risk without improving generalization.

---

#### Exp T-06 — Asymmetric EVT Penalty

**What changed:** The EVT tail loss was given an asymmetric penalty: under-predictions of extreme events (predicting lower than actual) were penalized at 2× the standard weight.

**Why it was tried:** Under-predicting extreme PM2.5 events is more dangerous from a public health perspective than over-predicting. The asymmetric penalty was designed to encode this domain-relevant preference.

**Result:** Val MAE 19.258, Test MAE **21.141**, RMSE 38.085. Marginal gain, training noisier.

**Observations:** Borderline improvement, domain-justified. Was retained as default briefly but later reverted — the marginal nature of the gain and noisier training made it not worth the complexity.

---

#### Exp T-07 — Persistence Residual

**What changed:** The model's output was reframed as a residual on top of the last observed PM2.5 value, rather than a direct absolute prediction. Formally: `final_prediction = model_output + y_last`, where `y_last` is the PM2.5 value at the last observed timestep for each node.

**Why it was tried:** PM2.5 exhibits strong short-term autocorrelation — the last observed value is already a powerful predictor, especially at H1. By anchoring the model to this baseline, the model only needs to learn the correction (delta), which is a smaller and more structured learning problem.

**What it does:** The model's head outputs a signed delta `Δ`. The training target is transformed to `Y - y_last` in normalized space, so the loss is computed on residuals. At inference, `y_last` is added back. This is an inductive bias: it tells the model that continuity of pollution levels is a strong prior.

**Result:**
| Metric | GT+GAT (no residual) | GT+GAT + Residual |
|---|---:|---:|
| Test MAE | 21.184 | **20.624** |
| Test RMSE | 38.074 | **37.729** |

**Observations:** Confirmed, large gain: Δ MAE = −0.560. Improvement concentrated at H1–H2 (near-horizon), where the last-value persistence prior is most informative. This became a permanent architectural component — it was never reverted.

---

### Session 2026-04-16 (s2) — Decoder and Input Experiments

---

#### Exp T-08 — Per-Horizon Prediction Heads

**What changed:** Instead of a shared 2-layer MLP generating all 6 horizon outputs from the same encoder summary, separate MLPs were trained for each of the 6 horizons.

**Why it was tried:** The hypothesis was that different horizons require different feature combinations. Sharing a single MLP forces all horizons to use the same representation transformation.

**What it does:** Six independent `fc1 + fc2` parameter sets, one per horizon step, each taking the same `(B, N, H)` encoder summary as input.

**Result:** Val MAE 19.151, Test MAE **21.208**, RMSE 38.346 — worse.

**Observations:** All six heads receive exactly the same encoder summary, so separate weights cannot add information — they can only refit training distribution. The encoder summary is the bottleneck, not the decoder capacity.

---

#### Exp T-09 — Cross-Attention Decoder

**What changed:** Instead of decoding from the single last-token encoder summary, horizon queries attended over the full T=24 encoder token sequence via cross-attention.

**Why it was tried:** The last-token bottleneck forces the entire temporal context into one vector. Cross-attention was intended to let each horizon step directly access the most relevant timesteps from the lookback window.

**What it does:** Six learnable query vectors (one per horizon step) attend over the full sequence of 24 encoder output tokens via scaled dot-product attention. Each query produces a context vector used to generate its horizon prediction.

**Result:** Val MAE 20.030 (noisy throughout), Test MAE **21.998**, RMSE 40.288 — significantly worse.

**Observations:** Training was highly unstable (val MAE spiking to 22–23 between adjacent epochs). The attention mechanism is harder to optimize with limited training data. Additionally, the persistence residual already anchors predictions to the last observed token, creating interference with cross-attention over the full sequence.

---

#### Exp T-10 — 48-Hour Input Window

**What changed:** Input window extended from 24h to 48h. A new preprocessed dataset `X_48.npy` was generated.

**Why it was tried:** 48h of lookback provides the model with two full daily cycles, which might help learn daily periodicity patterns and provide better context for longer-horizon predictions.

**Result:** Val MAE 19.233, Test MAE **21.084**, RMSE 38.563 — worse than 24h.

**Observations:** Hours 25–48 are informative noise for PM2.5 forecasting at this dataset and horizon. The model converged faster but shallower. 24h is the optimal input window for this dataset — information beyond 24h is more noise than signal.

---

#### Exp T-11 — EVT Asymmetric Penalty + Persistence Residual Combined

**What changed:** Both the asymmetric EVT penalty and the persistence residual were active simultaneously.

**Why it was tried:** To test whether combining two mechanisms that each improved the baseline would compound.

**Result:** Val MAE 18.910, Test MAE **20.938**, RMSE 37.926 — worse than residual alone.

**Observations:** The two mechanisms compete. The persistence residual already implicitly handles extreme-event behavior (because the last observed value is high when pollution is high). Adding asymmetric EVT pressure on top of the residual creates conflicting gradient signals, widening the val/test gap to 2.028 (vs 1.790 for residual alone).

---

### Session 2026-04-17 (s1) — Remaining Architecture Axes

---

#### Exp T-12 — Post-Temporal GAT

**What changed:** An additional GAT layer was added after the Transformer encoder, applied to the encoder's `(B, N, H)` node summaries. The pre-temporal GAT remained.

**Why it was tried:** After the Transformer compresses each node's 24h history into a summary, nodes have never interacted with each other temporally. The post-temporal GAT was designed to add one final round of spatial mixing on the already-temporally-encoded summaries.

**What it does:** A Pre-LN normalization followed by a GAT layer followed by an additive residual, inserted between the Transformer last token and the prediction head.

**Result:** Val 19.020, Test MAE **21.022**, RMSE 38.514 — worse.

**Observations:** Spatial mixing of Transformer output summaries destabilizes training on the 12-node graph (val MAE spiked 1–3 between adjacent epochs). Val/test gap widened to 2.002. The double-spatial structure (GAT before and after Transformer) caused over-smoothing.

---

#### Exp T-13 — Temporal Attention Head

**What changed:** The direct head was replaced with a per-horizon attention head that attends over the full T=24 Transformer output sequence (not just the last token).

**Why it was tried:** This is a third attempt at the "last-token bottleneck" hypothesis. Instead of decoding from the summary token alone, per-horizon soft attention over all 24 tokens was hypothesized to surface timestep-specific context for each horizon.

**What it does:** The Transformer returns its full `(B, N, T, H)` output. Six horizon scorers produce attention weights over T=24; each weighted sum is passed through a shared MLP to produce the horizon prediction.

**Result:** Val 19.338, Test MAE **21.353**, RMSE 38.690 — failed. Val/test gap 2.015.

**Observations:** Three full-sequence access attempts (cross-attention decoder, post-temporal GAT, temporal attention head) all failed. The Transformer's last token already encodes the full temporal context — the intermediate tokens do not carry additional useful predictive signal beyond what is already summarized.

---

#### Exp T-14 — t-24 Daily Anchor

**What changed:** A learnable gate `σ(t24_logit)` was added. The prediction became `model_delta + y_last + σ(logit) × y_{t−24}`, adding a gated contribution from the PM2.5 value 24 hours prior as a daily periodicity prior.

**Why it was tried:** PM2.5 in Beijing shows daily periodicity correlated with traffic and industrial cycles. The t-24 anchor was intended to provide an explicit daily cycle reference.

**What it does:** The model has an additional scalar parameter `t24_logit`. During forward pass, the PM2.5 value from exactly 24h ago is extracted from the input, passed through a sigmoid gate, and added as an additional residual term.

**Result:** Gate converged to ~0.073 (weak signal exists but small). Val 18.871, Test MAE **21.193**, RMSE 38.048. Val/test gap widened to 2.322 (worst seen up to this point).

**Observations:** Weak daily periodicity signal exists but the distributional shift between validation and test periods makes the anchor counterproductive. The t-1 persistence prior already captures the dominant autocorrelation; the daily component adds overfitting noise.

---

### Session 2026-04-17 (s2) — Loss Weighting and Oracle Diagnostic

---

#### Exp T-15 — Horizon-Aware Loss Weighting

**What changed:** The base MSE loss was given per-horizon weights `[1.0, 1.0, 1.0, 1.5, 2.0, 2.5]` (normalized to mean=1), upweighting H4–H6 to focus gradient pressure on far-horizon accuracy.

**Why it was tried:** The model showed a clear horizon-degradation pattern (H1 MAE ≈ 10, H6 MAE ≈ 29). Redirecting gradient to far horizons was hypothesized to reduce this degradation.

**Result:** Val MAE 18.835, Test MAE **20.889**, RMSE 38.121 — failed. All horizons degraded uniformly (H1 +0.175, H6 +0.386).

**Observations:** Upweighting H4–H6 in the loss redirects gradient pressure but cannot improve performance on steps the model has no information about. The horizon degradation is information-limited, not capacity-limited. This was the definitive closure of all loss-based attempts to improve far-horizon accuracy.

---

#### Exp T-16 — Oracle Future Meteorology (Diagnostic)

**What changed:** The model was given observed future meteorological conditions (temperature, pressure, dew point, rainfall, wind speed, and wind direction for all 6 forecast hours) as additional input to the prediction head.

**Why it was tried:** Far-horizon PM2.5 is strongly influenced by future wind conditions (which determine pollution transport). In real deployment, this information is unavailable, but providing it as an oracle allows us to quantify how much performance is limited by missing future meteorology.

**What it does:** A separate projection `Linear(21, hidden_dim)` in the prediction head takes the 21 future meteorological features (per horizon step, per node) and adds their projection to the combined representation before the MLP decoder. At inference with future met, the model sees where the wind will be blowing and how weather will evolve.

**Result:**
| Metric | Baseline | Oracle Future Met |
|---|---:|---:|
| Test MAE | 20.624 | **19.488** |
| Test RMSE | 37.729 | **35.087** |
| R² | 0.833 | 0.856 |

Per-horizon improvement:

| Horizon | Baseline MAE | Oracle MAE | Δ |
|---:|---:|---:|---:|
| H1 | 10.450 | 10.411 | −0.039 |
| H2 | 15.997 | 15.653 | −0.344 |
| H3 | 19.808 | 19.136 | −0.672 |
| H4 | 22.998 | 21.808 | −1.190 |
| H5 | 25.894 | 23.977 | −1.917 |
| H6 | 28.601 | 25.945 | −2.656 |

**Observations:** The improvement grows monotonically with horizon and is near-zero at H1 — the exact signature of information limitation. Future wind/meteorological conditions contain signal for far-horizon forecasting that a model working only from the lookback window cannot access. This experiment is not deployable, but serves as a quantitative upper bound on what future meteorological information could provide for this architecture. An operational deployment using NWP weather forecast inputs could approximate this bound.

**Important caveat:** This oracle bound is specific to the meteorological axis of this architecture. It does not constitute the problem's performance ceiling — other architectural improvements or different data sources could improve performance beyond 19.488 independently.

---

### Session 2026-04-17 (s3) — Graph and Temporal Variants

---

#### Exp T-17 — Per-Timestep Dynamic Adjacency

**What changed:** Instead of aggregating the 24h wind history into a single adjacency matrix per sample, a separate adjacency matrix was computed for each of the 24 timesteps, producing a `(B, T, N, N)` tensor.

**Why it was tried:** The standard approach uses a single recency-weighted average of wind over the lookback window. Per-timestep adjacency preserves the full temporal evolution of wind conditions.

**Result:** Test MAE **20.625**, RMSE 37.731 — statistical tie (+0.001 vs 20.624).

**Observations:** The Transformer encoder already recovers temporal wind dynamics from the feature sequence (wspm and wind direction are explicit input features at every timestep). Per-timestep adjacency resolution is redundant given the temporal attention mechanism.

---

#### Exp T-18 — Transport-Time Weighted Adjacency

**What changed:** The wind adjacency weights were modulated by a Gaussian gate based on estimated transit time between stations: `exp(−(d/(u·3.6) − H_ref)²/σ²)` where `H_ref = 3.5h` is the reference transit time.

**Why it was tried:** Pollution transport between stations takes time proportional to distance and wind speed. Weighting adjacency by expected transit time adds a physically meaningful lag structure.

**Result:** Test MAE **20.624**, RMSE 37.728 — statistical tie (+0.000).

**Observations:** Identical results to baseline to three decimal places. The existing recency-weighted wind aggregation already captures the relevant temporal dynamics implicitly.

---

#### Exp T-19 — Multi-Scale Temporal Branch

**What changed:** A parallel short-horizon Transformer branch (1 layer, last-6-timestep window) was run alongside the main 24-step Transformer, with outputs fused via a learned sigmoid gate.

**Why it was tried:** The hypothesis was that very recent timesteps (last 6h) contain stronger short-term signals while the full 24h branch captures longer patterns. Multi-scale architectures are common in time-series forecasting.

**Result:** Test MAE **20.952**, RMSE 38.328 — worse (+0.328).

**Observations:** The global Transformer already attends to recent tokens through its self-attention mechanism. The local branch is redundant and adds slight overfitting from extra parameters.

---

#### Exp T-20 — Gaussian Plume Adjacency

**What changed:** The wind transport adjacency was replaced with a Gaussian plume dispersion kernel: crosswind Gaussian `exp(−d_cross²/2σ²)` × along-wind exponential `exp(−d_along/(u·τ))`, with upwind → neutral fallback.

**Why it was tried:** Atmospheric dispersion follows Gaussian plume physics — pollution spreads perpendicular to wind direction with Gaussian cross-sectional distribution. This is a more physically grounded model than the existing directional transport approximation.

**Result:** Test MAE **20.640**, RMSE 37.769 — statistical tie (+0.016).

**Observations:** Physics-principled graph construction produces identical generalization to the simpler transport approximation. The graph construction method is not the bottleneck; the spatial encoder's ability to exploit adjacency structure is.

---

#### Exp T-21 — Learned Meteorological Forecaster (MetForecaster)

**What changed:** A separate model (`MeteorologicalForecaster`) was trained to predict future meteorological conditions from the lookback window. These predicted future conditions were then fed into the oracle future-met architecture as a realistic substitute for observed future met.

**Why it was tried:** If the oracle experiment showed a ~1.1 MAE improvement with perfect future met, a good forecaster for future met could unlock some of that gain in a deployable system.

**What it does:** The MetForecaster mirrors the main model's architecture (GAT + Transformer), takes `(B, 24, 12, 21)` meteorological features as input, and outputs `(B, 6, 12, 21)` future met predictions. It is pre-trained separately on an MSE loss, frozen, and used to generate Z-predictions that are injected into the main model in place of observed future met.

**Result:** MetForecaster pre-training val MSE: 0.040. Main model Test MAE: **20.514** vs baseline 20.624 → Δ = −0.110 → **statistical tie**.

**Observations:** The MetForecaster predictions are too noisy to exploit the oracle signal. The PM2.5 model trained on noisy predicted met does not generalize better than having no future met at all. The oracle gap is information-theoretic — it requires NWP-quality meteorological forecasts to be exploitable.

---

### Session 2026-04-18 — Meteorology Experiments Concluded

---

#### Exp T-22 — Persistence Future Meteorology

**What changed:** Instead of a learned forecaster, future meteorological conditions were approximated by repeating the last observed values for all 6 forecast hours.

**Why it was tried:** A simpler and cheaper baseline to check whether any future-met approximation could help, without the complexity of a pre-trained forecaster.

**Result:** Test MAE **20.765** — worse than baseline (20.624).

**Observations:** Repeating the last-observed met creates an artificial flat temporal trajectory over 6h that conflicts with what the 24h lookback already encodes. The model is trained to handle dynamic meteorological sequences and finds the persistence approximation misleading rather than helpful.

---

#### Exp T-23 — Horizon-Weighted Persistence Residual

**What changed:** Instead of uniformly adding `y_last` to all horizon predictions, per-horizon learnable scalar weights `σ(logit_h)` were applied: `final_h = model_delta_h + σ(logit_h) × y_last`.

**Why it was tried:** The persistence prior should be more relevant at H1 (where current PM2.5 is a strong predictor) and less relevant at H6 (where the model should rely on learned patterns). Allowing the model to learn per-horizon persistence weights could reflect this naturally.

**Result:** Test MAE **21.026** — worse than uniform residual (20.624) by +0.401.

**Observations:** Learned weights overfit to the validation period distribution. The uniform weight of 1.0 is already optimal — any flexibility the model gains in weighting the persistence prior is used to fit validation-specific patterns that do not transfer to the test period.

---

## Phase 2 — Normalization and Feature Engineering

### Session 2026-04-22

---

#### Exp T-24 — Log1p Transform on PM2.5

**What changed:** `np.log1p` was applied to the PM2.5 target (index 0) and corresponding input feature before scaling.

**Why it was tried:** PM2.5 is strongly right-skewed: mean ~84.6, std ~92.4, max ~835 µg/m³. The original MinMaxScaler compressed 98% of training samples below 0.1 on [0,1], causing massive distribution mismatch. Log1p compresses the heavy tail and brings the distribution closer to a normal shape that gradient-based optimizers handle well.

**What it does:** Before fitting the scaler, `log1p(pm2.5)` is applied to the target array. After inference, `expm1` is applied to inverse-transform predictions back to µg/m³ before computing metrics.

**Result:**
| Metric | Baseline | +Log1p PM2.5 |
|---|---:|---:|
| Test MAE | 20.624 | **20.055** |
| Test RMSE | 37.729 | 38.572 |

**Observations:** MAE improved by −0.570 but RMSE regressed (+0.843). The reason: co-pollutant inputs (PM10, SO2, NO2, CO, O3) remained in their original MinMax space — inconsistent normalization between correlated features degraded extreme-event representation.

---

#### Exp T-25 — Log1p Transform on PM2.5 + All Pollutants

**What changed:** Log1p extended to all correlated pollutant inputs: PM10, SO2, NO2, CO, O3 (indices 1–5) in addition to PM2.5 (index 0).

**Why it was tried:** PM2.5, PM10, SO2, NO2, CO, and O3 are all right-skewed and spatially correlated. Applying log1p consistently across all of them ensures the encoder sees harmonized representations for related features.

**Result:**
| Metric | Baseline | Log1p PM2.5 only | Log1p all pollutants |
|---|---:|---:|---:|
| Test MAE | 20.624 | 20.055 | **19.845** |
| Test RMSE | 37.729 | 38.572 | **37.780** |

**Observations:** Both MAE and RMSE improved. RMSE fully recovered from the regression in Exp T-24. The cumulative gain from baseline: Δ MAE = −0.779 — clearly above the noise floor and a confirmed, substantial improvement. Val/test gap also tightened (1.984 vs 2.239 in T-24).

**Thesis significance:** A preprocessing change alone, without any architecture modification, produced nearly the same gain as the entire architecture search. This was a strong finding that the prior normalization was the dominant performance limiter for all 15+ earlier architecture experiments.

---

#### Exp T-26 — Global StandardScaler (replacing MinMaxScaler)

**What changed:** The MinMaxScaler was replaced with a `StandardScaler` (zero mean, unit variance) applied after log1p transformation.

**Why it was tried:** MinMax scaling is sensitive to outliers — extreme PM2.5 events compress the majority of the distribution. StandardScaler distributes the training signal more evenly around the mean.

**Result:**
| Metric | Log1p + MinMax | Log1p + StdScaler |
|---|---:|---:|
| Test MAE | 19.845 | **19.813** |
| Test RMSE | 37.780 | **37.507** |

**Observations:** Statistical tie on MAE (−0.032), mild improvement on RMSE (−0.273). Alpha gate also shifted from 0.64 to 0.33 due to changed gradient scaling. Retained as default because RMSE improved and the principled normalization is preferable. This became the canonical scaler for all subsequent experiments.

---

#### Exp T-27 — Per-Station StandardScaler

**What changed:** Instead of a single global StandardScaler for the PM2.5 target, 12 independent per-station scalers were fitted to each station's training PM2.5 distribution.

**Why it was tried:** Different monitoring stations have different baseline concentration levels. Per-station normalization could reduce systematic bias in the loss function across stations with very different typical values.

**Result:**
| Metric | Global Std | Per-Station Std |
|---|---:|---:|
| Test MAE | 19.813 | 19.845 |
| Test RMSE | 37.507 | 37.566 |

**Observations:** Complete tie on all metrics (within noise). Rejected. Global scaler retained — simpler, same accuracy.

---

#### Exp T-28 — Rain Log1p Transform

**What changed:** Log1p was additionally applied to the rainfall feature (index 9).

**Why it was tried:** Rain is zero-inflated (most hours have zero rainfall) and right-skewed on non-zero days. It was a natural extension of the pollutant log1p treatment.

**Result:** Test MAE **19.942**, RMSE 37.741 — worse (+0.129 MAE).

**Observations:** Rain is zero-inflated — log1p(0) = 0 and log1p of small rainfall values compresses the already-low variance, making the distinction between light-rain and no-rain events less distinguishable to the model. Rejected.

---

#### Exp T-29 — PM2.5 First-Difference (Delta) Feature

**What changed:** An explicit PM2.5 rate-of-change feature (first-difference across timesteps) was added as an additional input channel at index 17. The wind direction indices shifted accordingly.

**Why it was tried:** The rate of change of PM2.5 provides directional information (is pollution rising or falling?) that may help distinguish persistence events from trend reversals.

**Result:** Val MAE 17.992 (misleadingly low), Test MAE **20.119**, RMSE 38.440 — worse. Val/test gap widened from 1.592 to 2.127.

**Observations:** The Transformer encoder already derives rate-of-change information implicitly from the 24h PM2.5 sequence through its self-attention mechanism. Explicit delta feature is redundant and adds gradient noise, overfitting the validation period.

---

#### Exp T-30 — Chinese Holiday Indicator

**What changed:** A binary holiday indicator was added as an input feature, marking major Chinese holidays (Spring Festival ±5 days, Golden Week Oct 1–7, Labour Day May 1–3).

**Why it was tried:** Holiday periods in Beijing are associated with fireworks (high PM2.5 spikes during Spring Festival) and traffic pattern changes. Encoding holiday context was expected to help the model anticipate these events.

**Result:** Val MAE 18.400, Test MAE **20.372**, RMSE 39.133 — worse (+0.559 MAE).

**Observations:** The model receives only the holiday status during the past 24h lookback window. The actual pollution spikes happen at the same time as fireworks, meaning the model needs future holiday status to anticipate spikes. Past-window holiday flags are largely redundant with the existing month/day cyclical encodings. Rejected.

---

## Phase 3 — Training Optimization and Adjacency Variants

### Session 2026-04-23

---

#### Exp T-31 — FFN Dimension 4× (256 vs 128)

**What changed:** The Transformer encoder's feed-forward network dimension was increased from `2×hidden_dim` (128) to `4×hidden_dim` (256), matching the standard Transformer FFN ratio.

**Why it was tried:** The default Transformer architecture uses 4× FFN expansion. The current 2× choice was a conservative capacity decision made early in the project.

**Result:** Val MAE 18.047 (misleadingly low at epoch 5), Test MAE **20.051**, RMSE 38.101 — worse.

**Observations:** Overfitting. The compact 2× FFN is appropriate for this dataset size (12 nodes, 24K training samples). Standard Transformer capacity ratios are designed for much larger datasets.

---

#### Exp T-32 — LR Scheduler: Monitor Val MAE Instead of Val Loss

**What changed:** The `ReduceLROnPlateau` scheduler's monitored metric was changed from `val_loss` (EVT hybrid loss) to `val_mae` (PM2.5 MAE in µg/m³).

**Why it was tried:** EVT hybrid loss value fluctuates depending on the composition of extreme-value events in each validation batch — it is not monotonically correlated with prediction quality. Val MAE is stable and directly reflects the metric of interest.

**Result:** Test MAE **19.813**, RMSE 37.507 — statistical tie.

**Observations:** No measurable gain at single seed, but the change is principled and makes the scheduler's signal more meaningful and interpretable. Retained as the permanent default.

---

#### Exp T-33 — Cosine Annealing with Linear Warmup

**What changed:** The ReduceLROnPlateau scheduler was replaced with a cosine annealing schedule with 5-epoch linear warmup and minimum LR ratio of 0.05.

**Why it was tried:** Cosine annealing is widely used in Transformer training and avoids the stochastic behavior of plateau detection.

**Result:** Test MAE **20.115**, RMSE 37.925 — worse.

**Observations:** ReduceLROnPlateau holds the LR high while the model is still improving; cosine annealing decays monotonically regardless of progress. For a model that converges fast and early-stops at epoch ~7, the adaptive scheduler is a better fit. Reverted.

---

#### Exp T-34 — Gaussian Input Noise Augmentation

**What changed:** During training only, Gaussian noise `N(0, 0.02)` was added to continuous input features (indices 0–16) to regularize the encoder.

**Why it was tried:** Input noise augmentation is a standard regularization technique that discourages the model from memorizing specific input patterns.

**Result:** Val MAE 18.23 (unchanged), Test MAE **19.927**, RMSE 37.687 — worse.

**Observations:** Noise degraded signal without improving generalization. The val/test gap was unchanged, meaning the regularization did not address the distribution-shift-driven gap. Reverted.

---

#### Exp T-35 — Correlation-Based Static Adjacency

**What changed:** The Gaussian distance decay component of the adjacency (`A_dist`) was replaced with Pearson correlation of PM2.5 between station pairs, computed on training data. The wind component was unchanged.

**Why it was tried:** Stations with high PM2.5 co-movement may share emission sources or transport pathways beyond what simple geographic distance captures. Correlation encodes long-run spatial PM2.5 relationships.

**Result:** Val MAE 18.220, Test MAE **19.831**, RMSE 37.565 — statistical tie (+0.018 vs 19.813). Alpha collapsed: 0.33 → 0.21.

**Observations:** Alpha collapse occurred because the correlation matrix is pre-normalized (row sums ≈ 1) whereas the original A_dist has row sums ~2–3. This scale difference destabilized the alpha gate. Long-run PM2.5 co-movement is already captured by node embeddings and shared temporal encoder patterns.

---

#### Exp T-36 — Learnable Static Adjacency

**What changed:** The fixed Gaussian distance decay matrix was replaced with 144 trainable logit parameters (12×12), initialized from the distance decay values. The sigmoid-activated and row-normalized result was used as the static adjacency component.

**Why it was tried:** If the Gaussian distance decay is suboptimal, end-to-end learnable adjacency could find a better spatial structure directly from PM2.5 prediction error.

**Result:** Val MAE 18.222 (epoch 7, noisy trajectory), Test MAE **19.836**, RMSE 37.575 — statistical tie (+0.023). Alpha collapsed: 0.620 → 0.213.

**Observations:** Same alpha-collapse pattern as correlation adjacency — any change to the static adjacency's gradient landscape destabilizes the learnable alpha gate. The Gaussian distance decay is already near-optimal as a spatial prior; gradient signal through the adjacency matrix is too weak for 144 logit parameters to find improvements.

**Pattern established:** Any modification to the `A_dist` component of the adjacency (not `A_wind`) that changes its scale or gradient landscape collapses the alpha gate. The alpha gate is a fragile parameter that stabilizes only under the baseline configuration.

---

### Session 2026-04-24 — Loss and Optimizer Experiments

---

#### Exp T-37 — Huber Loss + AdamW + wspm Log1p

**What changed:** Three simultaneous changes: (1) EVT base loss changed from MSE to Huber (δ=1.0), (2) optimizer changed from Adam to AdamW (weight decay λ=1e-5), (3) wspm (wind speed, index 10) added to log1p transform.

**Why it was tried:** Huber loss is more robust to outliers by switching from quadratic to linear loss for large errors. AdamW properly decouples weight decay from the adaptive gradient update. wspm log1p followed the same normalization logic as other right-skewed features.

**Result:** Val MAE 18.128 (better), Test MAE **19.921**, RMSE 38.216. Alpha collapsed: 0.648 → 0.116 (worst seen). Val/test gap widened.

**Critical finding:** wspm (index 10) is read directly by `build_dynamic_adjacency_gpu` as raw wind speed in m/s. Log1p transformation corrupts the wind adjacency computation — the adjacency builder interprets `log1p(wspm)` as a wind speed, producing incorrect transport weights. This feature must never be transformed.

---

#### Exp T-38 — Huber Loss + AdamW (without wspm log1p)

**What changed:** Same as T-37 but with wspm excluded from log1p transformation.

**Why it was tried:** To isolate whether the wspm corruption was the primary failure mode.

**Result:** Val MAE 17.979 (best seen), Test MAE **19.977**, RMSE 38.207. Alpha collapsed: 0.640 → 0.159. Val/test gap 1.998 (worst seen).

**Key finding on alpha collapse mechanism:** Huber loss switches from quadratic to linear gradient for large errors. This reduces the gradient magnitude from spatially distant high-error events, which is the primary signal driving the wind adjacency (`alpha_logit`) to stay active. With weakened large-error gradients, there is less incentive for the model to use the wind graph component — alpha collapses. The same mechanism applies to AdamW: weight decay on `alpha_logit` creates an additional pull toward zero.

**Forbidden operations established:** Huber loss, AdamW, and wspm log1p are permanently forbidden. Any loss that reduces the gradient magnitude for large spatial errors will collapse the alpha gate. EVT-MSE is the only stable loss configuration for this model.

---

#### Exp T-39 — Multi-Task Auxiliary Pollutant Heads

**What changed:** Five additional prediction heads were added to the model (one each for PM10, SO2, NO2, CO, O3), sharing the same encoder. An auxiliary MSE loss on future co-pollutant predictions was added at `λ_aux × MSE(aux_preds, Y_aux)`.

**Why it was tried:** Forcing the shared encoder to explain multiple correlated pollutants simultaneously could regularize the encoder representations and capture cross-pollutant emission/transport patterns that PM2.5-only supervision misses.

**What it does:** The model gains an `aux_head` (same MLP structure as the PM2.5 head) that predicts future PM10/SO2/NO2/CO/O3 from the same `(B, N, H)` encoder summary. The aux loss contributes ~10% of the total gradient. Early stopping and evaluation use PM2.5 MAE only.

**Result:** Test MAE **20.200** (baseline 19.793) — worse (+0.407).

**Observations:** PM10/SO2/NO2/CO/O3 signals are already implicitly encoded in the input features (indices 1–5). Auxiliary supervision adds gradient competition without providing new information. The additional heads pull the encoder away from PM2.5-optimal representations.

---

### Session 2026-04-25 (s1) — Residual and Structural Variants

---

#### Exp T-40 — Station × Horizon Output Bias

**What changed:** 72 zero-initialized learnable scalar parameters (12 stations × 6 horizons) were added as an additive bias to the model's output predictions in normalized space.

**Why it was tried:** Each station has different typical PM2.5 levels and different systematic prediction biases across horizons. A simple additive correction in the output space can absorb station-specific and horizon-specific biases without disturbing any intermediate representation.

**What it does:** After the prediction head generates `(B, 6, 12)` outputs, a learnable `(6, 12)` bias matrix is added. Starting from zeros, these parameters calibrate any remaining systematic offset per station per horizon.

**Result:** Test MAE **19.793**, RMSE 37.475 vs 19.813 — statistical tie (−0.020). Alpha unaffected (0.33).

**Observations:** Minimal measurable gain, but the mechanism is safe (zero-initialized, does not touch gradient path to alpha), theoretically sound (absorbs systematic biases), and adds only 72 parameters. Retained as permanent default.

---

#### Exp T-41 — Rolling-Mean Residual Anchor

**What changed:** The persistence residual anchor `y_last` was changed from the last single observed PM2.5 value to the mean of the last 6 observed values.

**Why it was tried:** The last single observation can be noisy. A rolling mean might provide a smoother and more stable baseline for the residual.

**Result:** Test MAE **21.094** — worse (+1.281 vs 19.793).

**Observations:** PM2.5 changes rapidly. Smoothing the anchor loses the model's ability to track rapid changes in pollution level — particularly harmful during spike onset and decay. The last-value anchor is the correct prior.

---

#### Exp T-42 — Trend Extrapolation Residual

**What changed:** The persistence prior was changed from flat `y_last` to a linear extrapolation: `y_last + h × slope`, where slope is estimated from the last 6h first-difference.

**Why it was tried:** If PM2.5 is trending upward or downward at the end of the lookback window, following that trend for the first few horizons might improve near-term accuracy.

**Result:** Test MAE **20.078** — worse (+0.265 vs 19.793).

**Observations:** PM2.5 frequently reverses direction near peaks and valleys. Trend extrapolation amplifies errors when the direction reverses — which happens commonly during pollution episodes. The prediction head already implicitly learns non-flat multi-step outputs. Flat last-value anchor is more robust.

---

#### Exp T-43 — Soft Regime Conditioning

**What changed:** A zero-initialized linear shortcut `Linear(1, hidden_dim)` was added from the last PM2.5 observation directly to the encoder output summary.

**Why it was tried:** The last PM2.5 value encodes the current pollution "regime" (clean vs polluted). An explicit shortcut could allow the decoder to condition on this regime without relying solely on the encoded sequence.

**Result:** Test MAE **20.118** — worse (+0.305 vs 19.793).

**Observations:** The last PM2.5 value is already input feature 0 in the 24h sequence — it is fully accessible to the Transformer encoder. Adding it again as a separate shortcut is redundant and disturbs encoder representations during training.

---

### Session 2026-04-25 (s2) — Distribution Shift Handling

---

#### Exp T-44 — RevIN (Reversible Instance Normalization)

**What changed:** Per-instance normalization was applied to the PM2.5 channel before the model forward pass, and reversed on the output. Each input window's PM2.5 is normalized by its own mean and standard deviation across the 24 timesteps.

**Why it was tried:** The persistent val/test gap (~1.6 MAE) suggests distribution shift between the training/validation period and test period. RevIN normalizes each sample to have zero mean and unit variance, removing instance-level distribution shifts.

**What it does:** Before the model sees the input, PM2.5 values across the 24-timestep window are mean-subtracted and variance-normalized per (sample, node). The model learns to predict in this normalized space. After inference, the normalization is reversed to recover absolute PM2.5 values.

**Result:** Val MAE **18.898** (worse), Test MAE **20.723**, RMSE 39.095 — clearly worse (+0.930 MAE). Alpha unchanged at 0.34 (no collapse).

**Root cause:** Dual-anchor conflict. The persistence residual anchors each prediction to `y_last` (last absolute value). RevIN anchors each prediction to the window mean. These two competing level corrections are inconsistent: `y_last ≠ window_mean`, creating systematic confusion about which "level" to predict residuals from. RevIN and persistence residual are fundamentally incompatible.

---

## Phase 4 — Hyperparameter Optimization

### Session 2026-05-01

---

#### Exp T-45 — Per-Node EVT Thresholds

**What changed:** The single global EVT tail threshold (90th percentile over all stations) was replaced with 12 per-station thresholds, one per monitoring station.

**Why it was tried:** Different monitoring stations have different PM2.5 distributions (e.g., Huairou is cleaner than Dongsi). Station-specific thresholds would target the local tail distribution of each station rather than a global average.

**Result:** Test MAE **19.775**, RMSE 37.428 vs baseline 19.793 / 37.475 — statistical tie (Δ −0.018).

**Observations:** Station PM2.5 tail distributions are similar enough that per-node thresholds add no measurable signal. Rejected.

---

#### Exp T-46 — Extended Early-Stopping Patience (25 vs 15)

**What changed:** Early stopping patience was increased from 15 to 25 epochs.

**Why it was tried:** The model's best checkpoint appeared consistently at epoch ~7. More patience was intended to let the model escape local optima and find further improvements.

**Result:** Test MAE **19.799**, RMSE 37.481 — statistical tie. Best checkpoint was still at epoch 7. Epochs 8–32 never surpassed epoch 7.

**Observations:** The model converges fast and then oscillates without improvement. The learning rate is the limiting factor, not patience. Reverted to patience=15.

---

#### Exp T-47 — Optuna Hyperparameter Optimization (50 Trials)

**What changed:** A full hyperparameter search was run using Optuna with 50 trials. Search space included: hidden_dim ∈ {64,96,128}, learning_rate, weight_decay, batch_size, dropout, wind_alpha, distance_sigma, wind_recency_beta, evt_lambda, evt_xi, evt_tail_quantile.

**Why it was tried:** All hyperparameters except hidden_dim were inherited from the GCN-LSTM phase. The GraphTransformer+GAT architecture had never been tuned.

**Best trial:** Val MAE 17.963. Key parameters: LR=0.000143, wind_alpha=0.892, hidden_dim=96.

**Adopted (all Optuna params):** Test MAE **20.116**, RMSE 38.009 — worse. Val/test gap widened from 1.58 to 2.15.

**Adopted (all except LR, keep LR=1e-3):** Test MAE **20.846**, RMSE 40.196 — much worse. Alpha collapsed: 0.893 → 0.183.

**Key findings:**
1. **Parameter coupling:** The Optuna best-trial package is a coupled unit. `wind_alpha=0.892` requires `LR=0.000143` (7× lower than baseline) to avoid alpha collapse. High initial wind emphasis + high LR = large gradient steps that push alpha down before it can stabilize. Cherry-picking individual Optuna parameters without the full package is dangerous.
2. **Val overfitting:** Single-seed Optuna on a fixed 15% validation split selects for validation-period wind and seasonal patterns that do not generalize to the test period (different season). Val/test gap widening is a direct consequence.

**Conclusion:** No Optuna parameters adopted. Baseline config restored.

---

## Phase 5 — Late Architecture Experiments

### Sessions 2026-05-01 (s2) and (s3)

---

#### Exp T-48 — Edge-Conditioned GAT Values

**What changed:** The GAT value aggregation was modified to condition on the adjacency edge scalar: `value_ij = W_v(h_j) + W_edge(adj_ij)`, where `W_edge` is a linear map from the scalar edge weight to the feature space. `W_edge` was initialized to zero so the model starts identical to baseline.

**Why it was tried:** The adjacency scalar already serves as an attention bias, influencing which neighbors are attended to. Conditioning the value (what information is passed) on the edge weight could allow the model to modulate the content of messages based on edge strength, not just their weight.

**Result (spatial-first):** Test MAE **20.130**, RMSE 38.561 — worse (+0.337). Alpha dropped to 0.287.

**Observations:** The adjacency scalar is already fully exploited as an attention bias. Using it also to condition value aggregation introduces redundant and conflicting signal. The edge feature pathway learned to activate but degraded generalization.

---

#### Exp T-49 — Dilated TCN Parallel Branch

**What changed:** A 4-layer dilated Temporal Convolutional Network (dilations [1, 2, 4, 8], kernel size 3, receptive field 31h) was added as a parallel branch to the Transformer encoder. Outputs were fused additively with a learned gate initialized to 0.

**Why it was tried:** TCNs excel at capturing local temporal patterns via dilated convolutions. The hypothesis was that a TCN branch could complement the global self-attention of the Transformer with multi-scale local structure.

**Result:** Test MAE **20.153**, RMSE 38.021 — worse (+0.360). Added ~50K parameters. Best val checkpoint appeared at epoch 2.

**Observations:** +50K parameters for 12 nodes is a dramatic capacity overshoot. The TCN branch fits the validation distribution pattern within a few epochs then overfits. The Transformer's attention already handles temporal patterns effectively.

---

#### Exp T-50 — Dual-Channel Spatial (Separate Distance and Wind Streams)

**What changed:** Instead of the single mixed adjacency `A = (1−α)×A_dist + α×A_wind`, two independent GAT layers were run in parallel: one using only the distance adjacency, one using only the wind adjacency. Their outputs were summed additively.

**Why it was tried:** In the single-channel design, the learnable alpha gate can suppress the wind component. The dual-channel design prevents this — both spatial components always contribute, potentially preserving wind signal even when training would prefer to suppress it.

**What it does:** `h_dist = GAT_dist(x, A_dist)` and `h_wind = GAT_wind(x, A_wind)` run independently. The encoder output is `x + h_dist + h_wind`.

**Result:** Val MAE 18.056 (epoch 12), Test MAE **20.405**, RMSE 39.305 — worse (+0.612 MAE, +1.830 RMSE).

**Observations:** The calibrated alpha-mixed hybrid is already near-optimal. Forcing additive two-stream aggregation means neither channel can be suppressed when suppression is appropriate. Distance smoothing and directional transport priors sometimes conflict, and the Transformer cannot resolve both cleanly. The alpha gate's learned weighting is load-bearing, not a liability.

---

### Sessions 2026-05-02

---

#### Exp T-51 — MC Dropout (All Layers at Test Time)

**What changed:** Dropout layers were kept active during test inference with 10 forward passes averaged.

**Why it was tried:** MC Dropout produces an uncertainty estimate (variance across passes) alongside the point prediction, and may improve point accuracy through implicit ensembling.

**Result:** Test MAE worse than baseline 19.793. No specific numbers recorded.

**Observations:** All-layer MC dropout with 10 passes degraded point prediction accuracy. Keeping multiple dropout layers active introduces meaningful prediction variance that outweighs the ensemble benefit at this pass count.

---

#### Exp T-52 — MC Dropout (Head Only, 30 Passes)

**What changed:** Dropout was applied only in the prediction head, not in the encoder. 30 passes were averaged.

**Result:** Test MAE 19.796 — statistical tie with baseline 19.793.

**Observations:** Point accuracy unchanged. However, the ensemble produces sensible uncertainty estimates with horizon-growing variance (wider uncertainty at H6 than H1), which is physically meaningful. Retained as an optional uncertainty diagnostic tool only — not used as the primary point predictor.

---

#### Exp T-53 — Probabilistic Gaussian NLL Output

**What changed:** The model was extended to output both a mean μ and a log-variance log(σ²) per prediction. Training used Gaussian negative log-likelihood: `0.5 × (log(2πσ²) + (y−μ)²/σ²)` instead of EVT-MSE hybrid.

**Why it was tried:** A fully probabilistic output provides calibrated uncertainty intervals, which are valuable for air quality policy decisions. NLL is a proper scoring rule that simultaneously penalizes incorrect means and poor uncertainty calibration.

**Result:** Val MAE 18.202 at epoch 3 (then degraded), early stop epoch 18, Test MAE **20.155**, RMSE 38.194. Alpha collapsed: 0.605 → 0.277.

**Observations:** NLL loss divides squared error by σ² — for high-variance predictions, large errors get downweighted in the gradient. This is structurally identical to Huber's linear-for-large-errors behavior and triggers the same alpha collapse mechanism. Additionally, the `log(σ²)` term incentivizes inflating σ as an easy loss reduction strategy rather than improving μ. Added to the forbidden operations list.

---

#### Exp T-54 — PM2.5-Only Spatial Path

**What changed:** A separate projection `pm25_proj = Linear(1, hidden_dim)` was added for the GAT's spatial aggregation step, while the Transformer encoder continued to receive the full 33-feature projection. The GAT residual was added onto the full-feature representation.

**Why it was tried:** Comparison with STGATN (same dataset, F=1 input), which reports RMSE ~35. The hypothesis was that mixing all 33 meteorological features into the GAT spatial aggregation introduces noise; a PM2.5-only spatial projection might yield cleaner spatial signal.

**Result:** Val MAE 18.398 (epoch 4), Test MAE **20.340**, RMSE 38.841 — worse (+0.547 MAE). Alpha: 0.512 → 0.288.

**Observations:** Meteorological features in the GAT input provide useful neighborhood context, not noise. The separate projection created a weaker gradient path through the spatial component. Two competing projection streams add parameters without a coherent inductive bias.

---

#### Exp T-55 — Temporal-First Ordering

**What changed:** The ordering of spatial and temporal processing was reversed. In the baseline (spatial-first), the GAT layer operates on raw projected features before the Transformer. In temporal-first, the Transformer runs first on per-node feature sequences, and the GAT operates on the resulting temporal summaries.

**Why it was tried:** STGATN architecture (same dataset) uses temporal encoding before spatial mixing. The previous `use_post_temporal_gat` experiment (T-12) failed because it added post-GAT on top of existing pre-GAT, creating double spatial processing. Temporal-first properly removes the pre-temporal GAT entirely — a genuinely distinct design.

**What it does:** Forward path: `input_proj → node_embed → [skip GAT] → Transformer → last-token → post_gat(h, adj) → residual → head`. The GAT now receives temporally-enriched node summaries `(B, N, H)` rather than raw projected features.

**Result:**
| Metric | Spatial-First (prev best) | Temporal-First |
|---|---:|---:|
| Test MAE | 19.793 | **19.489** |
| Test RMSE | 37.475 | **37.271** |
| MAPE | 36.727% | 38.565% |

**Observations:** Both MAE and RMSE improved. Gain is Δ MAE = −0.304, which is at the single-seed noise floor (0.3–0.5). MAPE regressed +1.84%, providing a counter-signal. Tentatively adopted as the new best, pending multi-seed confirmation.

---

#### Exp T-56 — SD-Calibrator: Spectral and Affine

**What changed:** A post-hoc spectral calibrator was applied to the temporal-first model's predictions after training. Two modes were tested: (1) Wiener filter on the time axis fitting a frequency-domain correction from validation predictions, (2) per-(horizon, station) affine correction `α·y + β` fitting validation mean and variance.

**Why it was tried:** The persistent val/test gap suggested systematic prediction biases that a calibration step could address.

**Result:**

| Mode | Test MAE | Δ | Test RMSE |
|---|---:|---:|---:|
| Uncalibrated | 19.489 | — | 37.271 |
| Spectral | 29.125 | +9.636 | 39.553 |
| Affine | 20.249 | +0.760 | 38.676 |

**Root cause:** The validation period (approximately July–September) and test period (approximately September–December) cover different seasons in the chronological 70/15/15 split. The Wiener filter's DC gain of 0.608 corrects for a large mean bias in the validation period that does not transfer to test. The affine correction's per-slot bias of −4.71 µg/m³ is season-specific and does not generalize.

**Finding:** Spectral calibration methods require that the calibration distribution (validation) closely approximates the deployment distribution (test). The chronological split deliberately violates this assumption. Both modes rejected.

---

### Session 2026-05-03

---

#### Exp T-57 — Geographic Embeddings

**What changed:** Station coordinates (latitude, longitude) were Fourier-encoded and passed through a 2-layer MLP, with the output fused into the node embeddings. Additionally, a normalized Haversine distance matrix was projected via a zero-initialized linear map to provide a geographic bias on the GAT attention logits.

**Why it was tried:** Node embeddings (learnable per-station vectors) provide implicit positional encoding. Explicit geographic coordinates could provide more grounded spatial priors, particularly useful when wind direction alone is insufficient.

**Result:** Test MAE **19.806**, RMSE 38.556 vs baseline 19.489 / 37.271 — worse (+0.317 MAE, +1.285 RMSE). Alpha collapsed: 0.52 → 0.28.

**Root cause:** The geographic distance bias was added to the GAT attention logits. Any additive modification of the attention scores (beyond the existing wind-adjacency bias) changes the effective attention landscape seen by the alpha gate gradient, destabilizing the wind signal. The coordinate encoder fused with node embeddings via MLP (without any attention-logit bias) is safe; it is the attention-logit injection that caused the collapse.

**New forbidden pattern established:** Any additive bias injected into GAT attention logits beyond the existing wind-adjacency bias will collapse alpha. This pattern was previously seen in correlation-adj and learnable-adj but is now formally documented.

---

#### Exp T-58 — Edge-Conditioned GAT Values (Temporal-First Position)

**What changed:** The same edge-conditioned value mechanism as T-48 was re-tried, but now in the post-temporal GAT position (on Transformer summaries) rather than the pre-temporal position.

**Why it was tried:** T-48 was tried in spatial-first (pre-temporal). Temporal-first provides richer `(B, N, H)` inputs to the GAT (containing temporally compressed context). The hypothesis was that edge conditioning might be more effective with richer node representations.

**Result:** Test MAE **19.455**, RMSE 37.274 vs temporal-first baseline 19.489 / 37.271 — statistical tie (Δ −0.034 MAE, Δ +0.003 RMSE). Alpha stable (~0.29).

**Observations:** The previous spatial-first rejection (+0.337 MAE) was genuine — the temporal-first position neutralized the penalty to a tie but found no useful signal either. Rejected.

---

#### Exp T-59 — GATv2 in Temporal-First Post-GAT Position

**What changed:** The post-temporal GAT was changed from GATv1 to GATv2 (dynamic attention formulation).

**Why it was tried:** GATv2 had tied GATv1 in the earlier spatial-first setting (T-04). With richer Transformer-encoded inputs, GATv2's dynamic attention (which mixes source and target features before scoring) might better capture complex cross-node interactions.

**Result:** Rejected. No measurable improvement over temporal-first GATv1 baseline. Reverted.

---

#### Exp T-60 — Transport Delay Cross-Attention Fusion

**What changed:** The estimated wind-transport delay between all station pairs was computed and used to extract the PM2.5 value at each station from the corresponding delayed timestep in the lookback window. These delayed observations were then fused into the encoder output via cross-attention.

**Why it was tried:** Pollutant transport from station A to station B takes time proportional to distance and wind speed. If a spike was observed at station A, it should appear at station B roughly `d/(u·3.6)` hours later. This module explicitly provided that temporally-delayed information.

**Result:** Test MAE **19.565**, RMSE 37.419, MAPE 36.209% vs baseline 19.489 / 37.271 / 38.565% — statistical tie (Δ +0.076 MAE, MAPE recovered −2.356%).

**Observations:** The MAE and RMSE slightly worsened while MAPE improved. The improvement in MAPE (low-concentration events) did not compensate for the slight regression in absolute error. Added complexity (cross-attention module) for zero measurable gain. Rejected.

---

#### Exp T-61 — TransAtt Decoder v1 (Zero-Init Out Projection, 4 Heads)

**What changed:** The direct multi-step MLP head was replaced with a Transformer-style attention decoder: horizon queries attend over the encoder output plus a sequence context, with a zero-initialized output projection to guarantee identity-at-initialization (the model starts as equivalent to the direct head).

**Why it was tried:** An attention decoder could theoretically exploit different parts of the encoded representation for each horizon step, rather than using the same linear transformation.

**Result:** Test MAE **20.169**, RMSE 38.726 — worse (+0.680). Alpha healthy (~0.285).

**Root cause:** Zero-initialized output projection (`out_proj`) means the entire decoder output is zero in epoch 1. The model produces zero predictions for the first epoch, causing the LR scheduler to reduce learning rate before the decoder has learned anything useful. This cold-start failure prevented the decoder from ever recovering. Identified as the Pre-LN / zero-start guarantee violation.

---

#### Exp T-62 — TransAtt Decoder v2 (Xavier Out Projection, 2 Heads)

**What changed:** The zero-initialized output projection was replaced with Xavier initialization. The number of decoder heads was reduced from 4 to 2.

**Why it was tried:** To fix the cold-start failure identified in T-61.

**Result:** Test MAE **19.544**, RMSE 37.253, MAPE 37.12% vs baseline 19.489 / 37.271 / 38.565% — statistical tie (Δ +0.055 MAE, RMSE virtually identical). Added +46K parameters.

**Observations:** The cold-start failure was resolved, but the attention decoder provides no measurable accuracy benefit over the direct MLP head. At the scale of this problem (12 nodes, 6-step horizon), a direct multi-step head is sufficient. Attention decoder added 46K parameters for zero measurable gain. Thesis finding: the direct head is appropriate for this problem scale.

---

#### Exp T-63 — 12-Hour Input Window

**What changed:** Input window reduced from 24h to 12h.

**Why it was tried:** 48h had already been rejected (T-10). 12h was tested to complete the ablation: 12h, 24h, and 48h lookback windows, with 24h already being best.

**Result:** Test MAE **19.589**, RMSE 37.634 vs 24h baseline 19.489 / 37.271 — worse. Alpha dropped further (~0.268).

**Observations:** 12h is insufficient for wind-based spatial modeling — the wind aggregation window is shorter, degrading adjacency quality. Val MAE was slightly better (17.80 vs 17.88), confirming slight overfitting without the daily-cycle temporal context. 24h confirmed as optimal.

---

#### Exp T-64 — Node-Specific Input Projection

**What changed:** The single shared linear projection `Linear(33, hidden_dim)` was replaced with 12 independent projections, one per station: `(N, H, F)` parameter tensor applied via `einsum('btnf,nhf->btnh', x, weight)`.

**Why it was tried:** Different monitoring stations have distinct pollution source profiles and meteorological environments. A station-specific projection could allow each station's encoder to learn a tailored feature weighting.

**Result:** Val MAE 18.042 (epoch 8), Test MAE **20.359**, RMSE 39.263, MAPE 35.948% — worse (+0.870 MAE, +1.992 RMSE). Alpha healthy (~0.297). Added +23,936 parameters (84,810 → 108,746).

**Observations:** Clear overfitting: val MAE 18.042 → test MAE 20.359, a val/test gap of 2.3 vs 1.9 for the baseline. With only 12 stations and ~24K training samples, 12 independent projection matrices learn station-specific patterns in the validation period that do not transfer to test. The 12-node graph does not have sufficient structural diversity to justify per-node input encoders.

---

## Summary Table: All Transformer-Phase Experiments

| ID | Experiment | Test MAE | Test RMSE | Verdict |
|---|---|---:|---:|---|
| T-01 | GraphTransformer + GCN | 21.686 | 39.077 | Neutral (faster) |
| T-02 | GraphTransformer + GAT v1 | **21.184** | **38.074** | ✓ Confirmed +0.502 |
| T-03 | 2-layer GAT | 21.526 | 38.547 | ✗ Over-smoothing |
| T-04 | GATv2 spatial | 21.170 | — | ~ Tie |
| T-05 | Iso-capacity h=96 | 21.142 | 38.458 | ~ Tie, RMSE worse |
| T-06 | Asymmetric EVT penalty | 21.141 | 38.085 | ~ Borderline |
| T-07 | Persistence residual | **20.624** | **37.729** | ✓ Confirmed +0.560 |
| T-08 | Per-horizon heads | 21.208 | 38.346 | ✗ |
| T-09 | Cross-attention decoder | 21.998 | 40.288 | ✗ Unstable |
| T-10 | 48h input window | 21.084 | 38.563 | ✗ |
| T-11 | EVT penalty + residual | 20.938 | 37.926 | ✗ Competing |
| T-12 | Post-temporal GAT | 21.022 | 38.514 | ✗ Over-smoothing |
| T-13 | Temporal attention head | 21.353 | 38.690 | ✗ |
| T-14 | t-24 daily anchor | 21.193 | 38.048 | ✗ Val overfit |
| T-15 | Horizon-aware loss | 20.889 | 38.121 | ✗ |
| T-16 | Oracle future met | 19.488 | 35.087 | Diagnostic only |
| T-17 | Per-timestep adj | 20.625 | 37.731 | ~ Tie |
| T-18 | Transport-time adj | 20.624 | 37.728 | ~ Tie |
| T-19 | Multi-scale temporal | 20.952 | 38.328 | ✗ |
| T-20 | Gaussian plume adj | 20.640 | 37.769 | ~ Tie |
| T-21 | MetForecaster | 20.514 | — | ~ Tie |
| T-22 | Persistence future met | 20.765 | — | ✗ |
| T-23 | Horizon-weighted residual | 21.026 | — | ✗ |
| T-24 | Log1p on PM2.5 | 20.055 | 38.572 | Partial (RMSE regressed) |
| T-25 | Log1p on all pollutants | **19.845** | **37.780** | ✓ Confirmed +0.779 |
| T-26 | Global StandardScaler | **19.813** | **37.507** | ~ Tie MAE, RMSE better |
| T-27 | Per-station StdScaler | 19.845 | 37.566 | ~ Tie |
| T-28 | Rain log1p | 19.942 | 37.741 | ✗ |
| T-29 | PM2.5 delta feature | 20.119 | 38.440 | ✗ Val overfit |
| T-30 | Holiday indicator | 20.372 | 39.133 | ✗ |
| T-31 | FFN dim 4× | 20.051 | 38.101 | ✗ Overfit |
| T-32 | LR → val_mae | 19.813 | 37.507 | ~ Tie (principled, kept) |
| T-33 | Cosine annealing | 20.115 | 37.925 | ✗ |
| T-34 | Gaussian noise | 19.927 | 37.687 | ✗ |
| T-35 | Correlation adjacency | 19.831 | 37.565 | ~ Tie, alpha collapse |
| T-36 | Learnable static adj | 19.836 | 37.575 | ~ Tie, alpha collapse |
| T-37 | Huber + AdamW + wspm log1p | 19.921 | 38.216 | ✗ Alpha collapse |
| T-38 | Huber + AdamW | 19.977 | 38.207 | ✗ Alpha collapse |
| T-39 | Multi-task aux heads | 20.200 | ~38 | ✗ |
| T-40 | Station × horizon bias | **19.793** | **37.475** | ~ Tie (kept) |
| T-41 | Rolling-mean residual | 21.094 | — | ✗ |
| T-42 | Trend extrapolation | 20.078 | — | ✗ |
| T-43 | Soft regime conditioning | 20.118 | — | ✗ |
| T-44 | RevIN | 20.723 | 39.095 | ✗ Dual-anchor conflict |
| T-45 | Per-node EVT thresholds | 19.775 | 37.428 | ~ Tie |
| T-46 | Patience = 25 | 19.799 | 37.481 | ~ Tie |
| T-47 | Optuna (50 trials) | 20.116 | 38.009 | ✗ Val overfit |
| T-48 | Edge-conditioned GAT (spatial-first) | 20.130 | 38.561 | ✗ |
| T-49 | TCN parallel branch | 20.153 | 38.021 | ✗ Overfit |
| T-50 | Dual-channel spatial | 20.405 | 39.305 | ✗ |
| T-51 | MC dropout all layers | >19.793 | — | ✗ |
| T-52 | MC dropout head only | 19.796 | — | ~ Tie (uncertainty only) |
| T-53 | Gaussian NLL output | 20.155 | 38.194 | ✗ Alpha collapse |
| T-54 | PM2.5-only spatial path | 20.340 | 38.841 | ✗ |
| T-55 | Temporal-first ordering | **19.489** | **37.271** | ~ Borderline (needs multi-seed) |
| T-56 | SD-Calibrator spectral | 29.125 | 39.553 | ✗ Seasonal mismatch |
| T-56 | SD-Calibrator affine | 20.249 | 38.676 | ✗ Seasonal mismatch |
| T-57 | Geo embeddings | 19.806 | 38.556 | ✗ Alpha collapse |
| T-58 | Edge features (temporal-first) | 19.455 | 37.274 | ~ Tie |
| T-59 | GATv2 (temporal-first) | — | — | ✗ |
| T-60 | Transport delay cross-attn | 19.565 | 37.419 | ~ Tie |
| T-61 | TransAtt decoder v1 | 20.169 | 38.726 | ✗ Cold-start failure |
| T-62 | TransAtt decoder v2 | 19.544 | 37.253 | ~ Tie, +46K params |
| T-63 | 12h input window | 19.589 | 37.634 | ✗ |
| T-64 | Node-specific projection | 20.359 | 39.263 | ✗ Val overfit |

**Legend:** ✓ = confirmed improvement above noise floor · ✗ = rejected (worse or meaningfully worse) · ~ = statistical tie · Baseline-shifted entries reflect the best deployable model at the time of the experiment.

---

## Confirmed Gains (Above Noise Floor)

| Change | Δ MAE | Source |
|---|---:|---|
| GCN → GAT spatial module | −0.502 | T-02 |
| Persistence residual | −0.560 | T-07 |
| Log1p on PM2.5 + co-pollutants | −0.779 | T-25 |
| Temporal-first ordering (borderline) | −0.304 | T-55 |

---

## Key System Constraints Discovered

1. **wspm (index 10) must never be log-transformed.** Wind speed is read raw by the adjacency builder. Transformation corrupts transport weights.
2. **Huber loss and AdamW are forbidden.** Both reduce gradient magnitude for large spatial errors, collapsing the learnable alpha gate.
3. **Gaussian NLL loss is forbidden.** Variance normalization has the same alpha-collapsing effect as Huber.
4. **No additive bias to GAT attention logits** beyond the existing wind-adjacency bias. Any such injection destabilizes alpha.
5. **RevIN is incompatible with the persistence residual.** Dual-anchor conflict: `y_last ≠ window_mean`.
6. **wind_alpha and learning_rate are coupled.** High initial wind_alpha (>0.6) requires proportionally lower LR to prevent alpha collapse during early training.

---

## Current Best Deployable Model

**Architecture:** `graph_transformer_gat_v1_residual_log1p_all_std_stationbias_temporal_first`

| Metric | Value |
|---|---:|
| Test MAE | 19.489 |
| Test RMSE | 37.271 |
| MAPE | 38.565% |
| Seed | 42 (single seed) |

**Status:** Borderline gain of Δ −0.304 MAE over spatial-first (19.793). MAPE regressed +1.84%. Multi-seed validation (seeds 0, 7, 123) is required to confirm whether this gain is real or within single-seed noise.

If the temporal-first gain does not hold across seeds, the fallback best remains:  
`graph_transformer_gat_v1_residual_log1p_all_std_stationbias` — MAE **19.793**, RMSE **37.475**.

---

## Final Architecture: Detailed Specification

**Model identifier:** `graph_transformer_gat_v1_residual_log1p_all_std_stationbias_temporal_first`  
**File:** `models/transformer_model.py` — class `GraphTransformerModel`  
**Trainable parameters:** 84,810  
**Training seed:** 42 (single seed; multi-seed validation pending)

---

### Overview of the Forward Pass

The model processes a 24-hour window of observations from 12 monitoring stations and directly predicts PM2.5 concentrations at all 12 stations for each of the next 6 hours. The full pipeline — from raw input to final prediction — consists of seven sequential stages: preprocessing, dynamic graph construction, input projection, temporal encoding, spatial refinement, multi-horizon decoding, and residual correction.

```
Raw input (B, 24, 12, 33)
    │
    ▼
[1] Preprocessing: log1p on PM2.5 + pollutants, then StandardScaler
    │
    ▼
[2] Dynamic wind-aware adjacency construction → A_batch (B, 12, 12)
    │
    ▼
[3] Input projection + node identity embeddings → (B, 24, 12, 64)
    │
    ▼  [GAT skipped — temporal-first ordering]
    │
[4] Transformer encoder (2 layers, 4 heads) → (B, 12, 64)   [last token per node]
    │
    ▼
[5] Post-temporal GAT (Pre-LN + GATv1 + residual) → (B, 12, 64)
    │
    ▼
[6] Direct multi-horizon head (step queries + MLP) → (B, 6, 12, 1)
    │
    ▼
[7] Station × horizon bias correction + persistence residual → (B, 6, 12)
```

---

### Stage 1 — Preprocessing and Normalization

**Log1p transformation (applied before fitting scalers):**

The six pollutant channels (PM2.5, PM10, SO2, NO2, CO, O3 at feature indices 0–5) are right-skewed with heavy tails. Before any scaling, `np.log1p` is applied to these channels in both the input tensor and the PM2.5 target:

```
X[:, :, :, 0:6] = log1p(X[:, :, :, 0:6])
Y = log1p(Y)
```

The remaining features — meteorological variables (indices 6–10), temporal cyclical encodings (11–16), and wind direction one-hot (17–32) — are not transformed. Wind speed at index 10 is explicitly excluded because it is read in raw m/s form by the dynamic adjacency builder; transforming it would corrupt the wind transport weights.

**StandardScaler (global, fit on training data only):**

After log1p, a single global `StandardScaler` is fit on the flattened training inputs and a separate global `StandardScaler` is fit on the flattened training PM2.5 targets. Both scalers are frozen after fitting; they are applied identically to validation and test data. This ensures no information leakage from future data.

At evaluation time, predictions are inverse-transformed via `expm1(scaler.inverse_transform(pred))`, and clipped at zero to prevent negative PM2.5 values.

---

### Stage 2 — Dynamic Wind-Aware Adjacency Construction

A separate adjacency matrix `A_batch` of shape `(B, 12, 12)` is constructed for every batch during training and inference. It is not stored or precomputed — it is rebuilt from the actual wind observations in each input window.

**Distance component `A_dist`:**

A symmetric Gaussian distance decay matrix is precomputed once from the 12 station coordinates (WGS-84 latitude/longitude, Haversine distance):

```
A_dist[i,j] = exp(−d(i,j)² / σ²)    σ = 1800 km²
```

Self-loops are included (`A_dist[i,i] = 1.0`). This captures geographic proximity: stations 10–20 km apart have strong static coupling, stations 80 km apart have near-zero weight.

**Wind component `A_wind`:**

A directed transport matrix is computed from the wind observations in each input window. For each sample in the batch, the 24h wind history is temporally aggregated using exponentially increasing weights (recent observations weighted more heavily, `recency_beta = 3.0`). Wind direction is aggregated via circular mean, weighted jointly by temporal recency and wind speed intensity.

For each ordered station pair (i, j), the transport weight is:

```
A_wind[i,j] = source_alignment(i→j) × (0.5 + 0.5 × receiving_alignment(j←i)) × A_dist[i,j]
```

where `source_alignment` measures how closely the wind at station i points toward station j (using cosine of the angular deviation between wind direction and the i→j bearing), and `receiving_alignment` measures whether the wind at station j is consistent with air arriving from station i. Both factors are modulated by `tanh(wind_speed / 5.0)`, which saturates at strong winds (~15 m/s) and approaches zero in calm conditions. This produces a physically meaningful directed graph: edges are strong when wind blows from source to target at both ends of the transport path.

**Alpha mixing:**

The final adjacency for batch sample b is:

```
A[b] = (1 − α) × A_dist  +  α × A_wind[b]
```

where `α = σ(alpha_logit)` is a learned scalar parameter initialized to `σ(logit(0.6)) = 0.6`. During training, `alpha_logit` receives gradients and converges to approximately 0.30 by the early-stopping checkpoint, indicating the model assigns roughly equal weight to geographic distance and wind-driven transport. The alpha gate is fragile: it collapses toward zero if the loss function reduces gradient magnitude for large spatial errors (as happens with Huber, NLL, or AdamW).

The result is row-normalized to produce a row-stochastic directed graph, then converted to a GPU tensor `(B, 12, 12)`.

---

### Stage 3 — Input Projection and Node Identity Embeddings

**Input projection:**

A single shared linear projection maps the 33-dimensional feature vector for each (batch, timestep, node) to the hidden dimension:

```python
input_proj: Linear(33 → 64)   # no bias suppression; standard init
```

Output: `(B, 24, 12, 64)`.

**Node identity embeddings:**

Each of the 12 stations is assigned a learnable identity vector of dimension 64, initialized from `N(0, 0.01)`. These are broadcast over all timesteps and added to the projected input:

```python
node_embed: nn.Embedding(12, 64)
x = x + node_embed(station_ids).unsqueeze(0).unsqueeze(0)   # (B, 24, 12, 64)
```

This allows the model to learn station-specific biases and characteristics (e.g., that Dingling is a suburban background station and Dongsi is an urban core station) without any geographic encoding.

**GAT step — skipped in temporal-first mode:**

Because `use_temporal_first=True`, the spatial GAT that normally follows input projection is completely skipped. The `(B, 24, 12, 64)` tensor proceeds directly to the Transformer encoder with no inter-node interaction yet.

---

### Stage 4 — Transformer Temporal Encoder

The Transformer encoder processes the 24-timestep feature sequence independently for each node, sharing weights across all 12 nodes. This is achieved by reshaping the batch:

```
(B, 24, 12, 64) → permute → (B, 12, 24, 64) → reshape → (B×12, 24, 64)
```

Each of the `B×12` sequences then goes through:

**Sinusoidal positional encoding:**

Standard sinusoidal PE is added to inject positional information about which of the 24 timesteps each token corresponds to. Dropout (p=0.1) is applied after PE injection.

```
PE[t, 2k]   = sin(t / 10000^(2k/64))
PE[t, 2k+1] = cos(t / 10000^(2k/64))
```

**Transformer encoder (2 layers, Pre-LN):**

Each of the 2 encoder layers applies:

1. **LayerNorm** on the input (Pre-LN convention — normalizes before the sublayer, not after)
2. **Multi-head self-attention** (4 heads, head dimension = 64/4 = 16): each of the 24 tokens attends over all other 24 tokens. Attention is bidirectional — there is no causal mask. The full temporal context (past 24h) is available to all positions simultaneously.
3. **Residual addition:** `x = x + dropout(attention_out)`
4. **LayerNorm**
5. **Feed-forward network:** Linear(64 → 128) → GELU → Dropout → Linear(128 → 64)
6. **Residual addition:** `x = x + dropout(ffn_out)`

A final LayerNorm is applied after the last encoder layer.

The FFN uses `hidden_dim × 2 = 128` (compact, not the standard 4× ratio). This was a deliberate choice: standard 4× FFN (256) was tried and caused overfitting on this dataset size (Exp T-31).

**Last-token extraction:**

Only the final timestep token is extracted as the node summary. The full 24-token sequence is discarded:

```
x_out = x_global[:, -1, :]         # (B×12, 64)
x_out = x_out.reshape(B, 12, 64)   # (B, 12, 64)
```

This is analogous to taking the final hidden state of an LSTM. The Transformer's self-attention has already integrated information from all 24 timesteps into this last token through its attention mechanism, so discarding the earlier tokens loses no information.

---

### Stage 5 — Post-Temporal Spatial GAT

This is the only point at which nodes exchange information with each other. The GAT operates on the compact `(B, 12, 64)` Transformer summaries, not on the raw 24-timestep sequences.

**Pre-LN normalization:**

```python
h_norm = LayerNorm(64)(enc_out)   # (B, 12, 64)
```

**GATv1 attention (4 heads):**

For each head `k` with head dimension `d = 64/4 = 16`:

1. Project node features: `Wh_i = W · h_i` and `Wh_j = W · h_j` (shared linear, no bias, shape `(12, 64)`)
2. Compute attention coefficient: `e_ij = LeakyReLU(a_k^T [Wh_i^k ∥ Wh_j^k])` where `a_k ∈ ℝ^{2d}` is a learned per-head attention vector
3. Inject wind-aware adjacency as an additive bias before softmax: `e_ij += A_batch[b, i, j]` when `A_batch[b, i, j] > 0`, else `e_ij = −∞` (hard mask for zero-adjacency pairs)
4. Normalize with softmax over neighbors: `α_ij = softmax_j(e_ij)`
5. Aggregate: `h_i^k = Σ_j α_ij · Wh_j^k`

Concatenate all 4 heads: `h_i = [h_i^1 ∥ h_i^2 ∥ h_i^3 ∥ h_i^4] ∈ ℝ^{64}`. Apply dropout (p=0.1) to attention weights. Add learned bias term.

The GATv1 formulation decomposes the attention score into independent source and target terms (`a_src^T Wh_i + a_tgt^T Wh_j`), which is computationally efficient and avoids the full `(B, N, N, H, 2D)` tensor required by the naive concatenation form.

**Residual addition:**

```python
enc_out = enc_out + gat_out   # (B, 12, 64)
```

The additive residual ensures the GAT layer cannot damage the Transformer summary: if the spatial aggregation provides no useful signal, the gradient will drive `gat_out` toward zero and the representation passes through unchanged.

---

### Stage 6 — Direct Multi-Horizon Prediction Head

The `DirectHorizonHead` produces all 6 horizon predictions simultaneously (no autoregression, no sequential error accumulation).

**Step queries:**

Six learnable query vectors `q_h ∈ ℝ^{64}`, one per forecast step, are initialized with Xavier uniform. These vectors encode horizon-specific information — the model learns that H1 requires a different transformation from the same encoder summary than H6.

**Vectorized MLP:**

```python
# Expand encoder summary over horizon dimension
final_h_exp = enc_out.unsqueeze(2).expand(-1, -1, 6, -1)   # (B, 12, 6, 64)

# Add horizon-specific query vector
queries = step_queries.view(1, 1, 6, 64)
combined = final_h_exp + queries                            # (B, 12, 6, 64)

# Flatten and apply shared 2-layer MLP
combined = combined.reshape(B×12×6, 64)
combined = LayerNorm(64)(combined)
combined = GELU(Linear(64 → 64)(combined))
combined = Dropout(0.1)(combined)
out = Linear(64 → 1)(combined)                             # (B×12×6, 1)

# Restore shape: (B, 6, 12, 1)
out = out.reshape(B, 12, 6, 1).permute(0, 2, 1, 3)
```

The MLP is shared across all horizons — only the additive step query differentiates the 6 outputs. This keeps parameter count low while giving each horizon its own linear subspace to operate in.

---

### Stage 7 — Station × Horizon Bias, Persistence Residual, and Output

These three post-processing operations are applied sequentially after the prediction head.

**Station × horizon output bias:**

```python
station_horizon_bias: nn.Parameter, shape (6, 12), zero-initialized
predictions = predictions + bias.unsqueeze(0).unsqueeze(-1)   # broadcast over B, output_dim
```

72 scalar parameters, one per (horizon step, station) pair. Zero-initialized so training starts identical to a model without bias. These parameters absorb any remaining systematic per-station and per-horizon prediction offsets in the normalized space. The alpha gate gradient path is unaffected because the bias is applied after the entire encoder-decoder stack.

**Persistence residual:**

The last observed PM2.5 value at each station is extracted from the scaled input (in log1p+StdScaler space):

```python
y_last = X_input[:, -1, :, 0]   # (B, 12) — last timestep, PM2.5 feature (index 0)
```

The model's prediction is added to this anchor:

```python
predictions = predictions + y_last.unsqueeze(1).expand_as(predictions)
```

This reframes what the model actually learns: instead of predicting the absolute PM2.5 level at t+h, the model predicts the signed delta from the last observed value. The persistence baseline (predicting `y_last` for all horizons) is the zero prediction for this formulation — the model only needs to learn the correction.

**Final output:**

After squeezing the output dimension, predictions have shape `(B, 6, 12)` in log1p+StdScaler space. The inverse transform `expm1(scaler.inverse_transform(predictions))` recovers the original µg/m³ scale.

---

### Loss Function: EVT Hybrid MSE-Tail Loss

The training loss combines a standard mean squared error on all predictions with an additional tail penalty targeting extreme pollution events above the 90th percentile.

**Base loss:**

```
L_base = MSE(predictions, targets)
```

**Tail threshold:**

The global 90th percentile of PM2.5 in the training set (in log1p+StdScaler space) serves as the EVT threshold: `θ = 1.2143`.

**EVT tail penalty:**

For all (batch, horizon, node) positions where the target exceeds θ:

```
target_excess = target - θ
excess_weight = 1 + ξ × (target_excess / mean_excess)    ξ = 0.10
excess_weight = clamp(excess_weight, min=1.0)
L_tail = mean(excess_weight × (prediction − target)²)
```

The excess weight function is derived from Generalized Pareto Distribution theory: observations further into the tail receive progressively higher loss weight. This forces the model to be more accurate during severe pollution episodes without changing the loss structure for normal observations.

**Total loss:**

```
L = L_base + λ × L_tail    λ = 0.05
```

Lambda was set to 0.05 — small enough that the EVT tail term does not dominate gradients (which would destabilize the alpha gate) but large enough to provide measurable tail pressure.

---

### Optimizer and Training Protocol

| Setting | Value |
|---|---|
| Optimizer | Adam (not AdamW — weight decay on alpha_logit causes collapse) |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Gradient clipping | max_norm = 1.0 |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping patience | 15 epochs |
| Early stopping metric | Validation MAE in µg/m³ (not EVT loss) |
| Checkpoint selection | Best validation MAE |
| LR scheduler | ReduceLROnPlateau on val_mae, factor=0.5, patience=5 |
| Dropout | 0.1 (all dropout layers) |
| Seed | 42 |
| Deterministic mode | False |

Early stopping at epoch 26 (best checkpoint: epoch 11, val MAE 17.880). The LR scheduler monitors `val_mae` rather than `val_loss` because the EVT loss value fluctuates with the proportion of tail-exceeding events in each validation batch, making it an unreliable signal for the scheduler.

---

### Parameter Count Breakdown

| Component | Parameters |
|---|---:|
| Input projection `Linear(33→64)` | 2,112 |
| Node identity embeddings `Embedding(12, 64)` | 768 |
| Transformer encoder (2 layers, 4 heads, FFN=128) | ~66,000 |
| Post-temporal GAT (GATv1, 4 heads, h=64) | ~8,300 |
| Direct horizon head (step queries + MLP) | ~4,300 |
| Station × horizon bias | 72 |
| Alpha logit (learnable scalar) | 1 |
| **Total** | **~84,810** |

The model is deliberately compact. The dataset has only 12 nodes and ~24K training samples; larger capacity reliably increased overfitting in all experiments that tried it.

---

### Architectural Design Decisions and Justifications

Each non-default architectural choice is supported by an experiment in the log above:

| Decision | Justification |
|---|---|
| GATv1 spatial module | +0.502 MAE over GCN (T-02); GATv2 tied GATv1 (T-04) |
| 1 GAT layer | 2-layer caused over-smoothing on 12-node graph (T-03) |
| Temporal-first ordering | −0.304 MAE over spatial-first (T-55); borderline, single-seed |
| Persistence residual | −0.560 MAE over no-residual (T-07) |
| Log1p on PM2.5 + pollutants | −0.779 MAE over no-log-transform (T-25) |
| Global StandardScaler | Tied MAE vs MinMax, RMSE improved (T-26) |
| Station × horizon bias | −0.020 MAE statistical tie; safe, kept (T-40) |
| Direct head (no cross-attention) | Cross-attn worse (T-09); TransAtt tied with +46K params (T-62) |
| FFN dim = 2× (128 vs 256) | 4× caused overfitting (T-31) |
| Adam not AdamW | AdamW collapses alpha gate (T-38) |
| EVT-MSE not Huber/NLL | Both collapse alpha via gradient magnitude reduction (T-37, T-53) |
| wind_alpha = 0.6 initial | Optuna coupling: changing this without adjusting LR collapses alpha (T-47) |
| 24h input window | 48h was worse (T-10); 12h was worse (T-63) |
