# Spatio-Temporal Air Quality Forecasting (GCN-LSTM + Multi-Head Attention)

Bachelor project for PM2.5 forecasting across 12 Beijing monitoring stations using a spatio-temporal graph neural architecture.

## 1. Executive Summary

This project predicts PM2.5 for the next 6 hours using the previous 24 hours of multivariate station data.

- Spatial relations are modeled with Graph Convolution inside a custom Graph-LSTM cell.
- Temporal dynamics are modeled with stacked recurrent layers in an encoder-decoder setup.
- A decoder-side Multi-Head Attention module selectively uses encoder history while forecasting.
- Output is node-wise PM2.5 forecasts for all stations.

Reported run performance (original scale):

- RMSE: 41.62
- MAE: 22.81
- MAPE: 64.67%
- R2: 0.7973

## 2. Problem Setup

- Target: PM2.5 concentration
- Nodes: 12 fixed stations
- Input horizon: 24 hours
- Forecast horizon: 6 hours
- Input features per node: 33
- Batch format in training:
  - X: (batch, 24, 12, 33)
  - Y: (batch, 6, 12)

## 3. Repository Structure

```
Bachelor/
|- preproccess.py          # Raw CSV -> cleaned tensor + scalers + metadata
|- train.py                # Training loop, checkpointing, evaluation
|- requirements.txt
|- data/
|  |- raw/                 # Beijing CSV files (per station)
|  |- processed/           # X.npy, Y.npy, adjacency.npy, scalers, predictions
|- models/
|  |- layers.py            # GraphConvolution, GraphLSTMCell, MultiHeadAttention
|  |- encoder.py           # GraphLSTMEncoder
|  |- decoder.py           # GraphLSTMDecoder
|  |- model.py             # Full GCNLSTMModel
|  |- checkpoints/         # Saved weights
|- utils/
|  |- graph.py             # Distance-based adjacency construction
|  |- window.py            # Sliding window generation
|  |- tester.py            # Full test evaluation (overall/horizon/station)
```

## 4. Data Pipeline

### 4.1 Preprocessing

`preproccess.py` performs:

1. Merge all station CSVs.
2. Missing value interpolation (station-wise), then bfill/ffill.
3. Cyclical time features (hour, month, weekday via sin/cos).
4. One-hot encoding for wind direction.
5. Tensorization into (T, N, F), with PM2.5 at feature index 0.
6. MinMax scaling (separate scaler for PM2.5).

Core feature ordering logic:

```python
target_col = ["pm2.5"]
pollutant_cols = ["pm10", "so2", "no2", "co", "o3"]
meteo_cols = ["temp", "pres", "dewp", "rain", "wspm"]
temporal_cols = ["hour_sin", "hour_cos", "month_sin", "month_cos", "weekday_sin", "weekday_cos"]
feature_cols = target_col + pollutant_cols + meteo_cols + temporal_cols + wind_cols
```

### 4.2 Graph Construction

`utils/graph.py` builds a static normalized adjacency from geographic distance:

```python
A[i, j] = np.exp(-d**2 / 100)
D = np.diag(np.sum(A, axis=1))
A_hat = D_inv_sqrt @ A @ D_inv_sqrt
```

### 4.3 Supervised Windows

`utils/window.py` converts tensor to learning windows:

```python
X.append(data[i:i+input_len])
Y.append(data[i+input_len:i+input_len+horizon, :, 0])
```

Why this design helps:

- The fixed 24h lookback gives enough context for daily PM2.5 behavior.
- The 6h target horizon keeps forecasting practical for operational use.
- Using PM2.5 at index 0 enforces a clean target path in both training and evaluation.

## 5. Model Architecture

### 5.1 High-Level Design

Encoder-Decoder with Graph-LSTM blocks and decoder-side temporal attention:

```
Input Sequence (24h, 12 nodes, 33 features)
	-> Encoder: 2 x GraphLSTM layers
	-> Decoder: Multi-Head Attention + 2 x GraphLSTM layers
	-> Output Projection
	-> Forecast (6h, 12 nodes, PM2.5)
```

### 5.2 Graph-LSTM Cell

Each recurrent step combines spatial and temporal processing:

```python
x_gcn = F.relu(self.gcn_i(x, adj))
h_gcn = F.relu(self.gcn_h(h, adj))
combined = torch.cat([x_gcn, h_gcn], dim=-1)
gates = self.gates(combined)
i, f, g, o = gates.chunk(4, dim=-1)
c_new = sigmoid(f) * c + sigmoid(i) * tanh(g)
h_new = sigmoid(o) * tanh(c_new)
```

### 5.3 Multi-Head Attention in Decoder

Attention is applied at every prediction step over encoder outputs:

```python
context, attn_weights = self.attention(
	 query=decoder_input,
	 key=encoder_outputs,
	 value=encoder_outputs
)
combined = torch.cat([decoder_input, context], dim=-1)
combined = self.context_proj(combined)
```

### 5.4 Component-by-Component Functionality and Benefit

1. Input projection layer:
	- Functionality: maps raw feature space (33) to hidden representation (64).
	- Benefit: gives a unified latent space where graph and temporal operators can work efficiently.

2. Positional encoding:
	- Functionality: injects timestep order information into encoder inputs.
	- Benefit: helps the model distinguish "same value at different times" (for example, morning vs night patterns).

3. GraphConvolution (spatial mixing):
	- Functionality: aggregates neighbor station information using normalized adjacency.
	- Benefit: captures cross-station pollution propagation patterns that single-station models miss.

4. GraphLSTMCell (spatio-temporal core):
	- Functionality: combines graph-transformed input and hidden state, then updates memory through LSTM gates.
	- Benefit: jointly models where pollution moves (space) and how it evolves (time) in one recurrent unit.

5. Stacked encoder layers (2 layers):
	- Functionality: first layer learns local dynamics; second layer refines higher-level abstractions.
	- Benefit: better representational power than a shallow encoder, while still lightweight.

6. Residual connections and layer normalization:
	- Functionality: skip connection in upper layers plus post-layer normalization.
	- Benefit: stabilizes optimization, improves gradient flow, and reduces training volatility.

7. Decoder multi-head attention:
	- Functionality: at each forecast step, attends over encoder history with multiple heads.
	- Benefit: each head can focus on different temporal cues (recent trend, periodic signal, anomaly context), improving forecast relevance.

8. Autoregressive decoder with teacher forcing:
	- Functionality: during training, sometimes uses ground truth as next input; during inference, uses own predictions.
	- Benefit: speeds and stabilizes training early, then gradually improves rollout robustness as teacher forcing decays.

9. Output projection:
	- Functionality: maps hidden decoder state to scalar PM2.5 per node.
	- Benefit: keeps output head simple and interpretable while preserving node-wise forecasting.

10. End-to-end objective (MSE in current baseline):
	 - Functionality: minimizes average squared prediction error over all horizons and nodes.
	 - Benefit: strong baseline objective for smooth regression, easy to compare with enhanced losses (e.g., EVT hybrid).

### 5.5 Why This Architecture Is Efficient for a Bachelor Project

- Parameter-efficient: 193,153 trainable parameters is relatively small for a graph + sequence model.
- Practical complexity: captures both spatial and temporal dependencies without requiring extremely deep networks.
- Interpretable upgrade path: each module can be upgraded independently (loss, adjacency, graph operator, preprocessing).
- Good research balance: sufficient sophistication for thesis contribution, but still reproducible on accessible hardware.

### 5.6 Training Hyperparameters (Current)

- hidden_dim: 64
- num_layers: 2
- num_heads: 4
- dropout: 0.1
- batch_size: 32
- learning_rate: 1e-3
- epochs: 100
- early stopping patience: 15
- teacher forcing: linear decay from 1.0 to 0.0

Total trainable parameters: 193,153

## 6. Training and Evaluation Workflow

### 6.1 Commands

1. Build processed tensor:

```bash
python preproccess.py
```

2. Build adjacency matrix:

```bash
cd utils
python graph.py
```

3. Build sliding windows:

```bash
cd utils
python window.py
```

4. Train model:

```bash
python train.py
```

4.1 Hyperparameter tuning with Optuna:

```bash
python optuna_tune.py --trials 30 --epochs 35 --patience 8
```

Notes:
- Tuning optimizes validation loss and disables test evaluation during trials (to avoid test leakage).
- Results are stored in `optuna_study.db` by default.
- You can resume the same study by reusing `--study-name` and `--storage`.

5. Evaluate best checkpoint:

```bash
cd utils
python tester.py
```

### 6.2 What `tester.py` Reports

- Overall metrics: RMSE, MAE, MAPE, R2
- Per-horizon metrics (+1h to +6h)
- Per-station metrics (12 stations)
- Saves test predictions and targets to `data/processed/`

## 7. Current Experimental Snapshot

From your reported run:

- Overall:
  - RMSE 41.6168
  - MAE 22.8055
  - MAPE 64.67%
  - R2 0.7973
- Horizon behavior: error increases as horizon increases (expected in autoregressive forecasting).
- Station behavior: heterogeneous difficulty across stations, indicating spatial complexity differences.

Interpretation:

- Model is strong for short-term forecasting (+1h to +3h).
- Performance degradation by +6h indicates room for stronger long-horizon handling.
- Architecture is already competitive for a lightweight 193k-parameter model.

## 8. Technical Strengths and Limitations

Strengths:

- Explicit spatio-temporal inductive bias (GCN + LSTM).
- Multi-head attention improves temporal context usage.
- Clean, reproducible preprocessing and evaluation scripts.
- Balanced parameter count vs predictive performance.

Limitations:

- Static adjacency does not encode dynamic wind transport physics.
- Standard regression loss tends to under-focus extreme spikes.
- Autoregressive decoding accumulates uncertainty toward +6h.

## 9. Next-Step Plan (Ranked by Efficiency Ratio)

Based on your architecture and the papers you listed, this is the recommended ranked roadmap (gain vs added computational cost):

| Rank | Solution | Source Paper(s) | Mechanism | Efficiency Ratio |
|---|---|---|---|---|
| 1 | Extreme Value (EVT) Hybrid Loss | A16 / A15 | Hybrid loss (MSE + Generalized Pareto tail-aware component) to focus spikes | Infinite (near-zero added model cost) |
| 2 | Directed Wind-Aware Adjacency | A18 / A13 | Dynamic directed adjacency using wind speed/direction | Very High |
| 3 | Graph Attention (GAT) Layers | A4 / A21 | Replace fixed GCN mixing with learned neighbor attention | High |
| 4 | Direct Multi-Horizon Decoding | Architecture-level upgrade | Predict all future steps jointly instead of recursively feeding predictions back | High |
| 5 | VMD / CEEMDAN Preprocessing | A6 / A21 | Decompose signal into trend/periodic/noise components | Moderate |

### 9.1 Why This Ranking Fits This Codebase

1. EVT Hybrid Loss (Rank 1):
	- Directly addresses your most expensive errors: pollution spikes.
	- Can be added in `train.py` loss computation without changing model topology.
	- Best immediate return for almost no parameter increase.

2. Directed Wind-Aware Adjacency (Rank 2):
	- Your current `utils/graph.py` adjacency is distance-only and static.
	- You already have wind features in the 33-dim input, so this is a physics-informed upgrade with low overhead.

3. GAT Upgrade (Rank 3):
	- In `models/layers.py`, replacing `GraphConvolution` with attention-based neighbor weighting can improve heterogeneous station modeling.
	- Adds moderate complexity and parameters.

4. Direct Multi-Horizon Decoding (Rank 4):
	- Your current decoder is autoregressive, so each predicted step becomes input to the next one.
	- That means uncertainty accumulates, which is exactly why error grows toward +6h.
	- A direct multi-horizon decoder predicts +1 to +6 jointly from the encoder representation instead of recursively depending on its own previous output.
	- This is the most targeted architectural fix for long-horizon degradation in your current model.

5. CEEMDAN/VMD (Rank 5):
	- Often effective for noisy PM2.5 signals, but adds preprocessing latency and pipeline complexity.
	- Better as a controlled extension after loss/adjacency upgrades.

### 9.2 Recommended Implementation Order

Practical sequence for your next experimental cycle:

1. Add EVT hybrid loss first.
2. Add directed wind-aware dynamic adjacency second.
3. Add direct multi-horizon decoding third, especially if +6h degradation remains the main weakness.
4. Re-run baseline vs upgraded comparison with same splits.
5. Only then evaluate GAT and decomposition if additional gains are needed.

This sequence gives the highest probability of measurable improvements without exploding training cost.

## 10. Advisor-Ready Thesis Positioning

If presenting this project, position it as:

"A lightweight, interpretable spatio-temporal graph forecasting framework that balances physical priors (graph structure), sequence modeling (LSTM), and adaptive context selection (attention), with a clear roadmap toward extreme-event robustness and transport-aware graph dynamics."

## 11. Environment

Dependencies in `requirements.txt`:

- torch >= 2.0
- numpy >= 1.24
- pandas >= 2.0
- scikit-learn >= 1.3
- joblib >= 1.3

---

If needed, I can add a second concise "one-page presentation slide version" section at the top with only: objective, method, key result, and next steps.