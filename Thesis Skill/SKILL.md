---
name: thesis-writer
description: >
  A thesis writing engine calibrated on a GUC bachelor's thesis in deep learning /
  spatiotemporal modeling. Use this skill whenever the user asks to write, draft, generate,
  expand, or complete a bachelor thesis, academic thesis, or research paper in any technical
  or engineering domain. Triggers include: "write my thesis", "draft a thesis chapter",
  "generate a thesis from my results", "help me write up my research", "turn my work into
  a thesis", or any mention of needing a structured academic document with Introduction /
  Background / Methodology / Results / Conclusion chapters. Also use when the user provides
  experimental results, a model description, or a dataset and asks for a full writeup.
  Works for any ML / deep learning domain: air quality forecasting, traffic prediction,
  NLP, computer vision, recommender systems, medical imaging, and more.
---

# Thesis Writer

This skill produces a complete, publication-quality GUC bachelor thesis from structured
research inputs. The blueprint is **fully domain-agnostic** — it encodes structure, flow,
argumentation patterns, and writing style only. All domain-specific terms are written as
`[PLACEHOLDER]` variables that get resolved from the user's actual research during generation.

---

## Step 0 — Terminology Map

Before writing any prose, derive and confirm this mapping from the user's inputs.
Ask for anything that cannot be inferred.

| Placeholder | Meaning | Example: PM2.5 thesis |
|---|---|---|
| `[DOMAIN]` | Application field | air quality forecasting |
| `[TARGET_VARIABLE]` | What is predicted | PM2.5 concentration |
| `[SPATIAL_UNIT]` | One graph node | monitoring station |
| `[SPATIAL_UNITS]` | Plural | monitoring stations |
| `[EDGE_TYPE]` | What graph edges represent | spatial proximity / wind correlation |
| `[INPUT_SIGNAL]` | Primary input feature(s) | hourly pollutant readings |
| `[DATASET_NAME]` | Dataset name | UCI Beijing Multi-Site Air Quality |
| `[DATASET_SOURCE]` | Dataset origin | UCI Machine Learning Repository |
| `[BASELINE_MODEL]` | Comparison model | LSTM / GRU / Informer |
| `[MODEL_FAMILY]` | Proposed model family name | GCN-LSTM with multi-head attention |
| `[VARIANT_1]` | First proposed model | GCN-LSTM (base) |
| `[VARIANT_2]` | Second proposed model | GCN-LSTM + multi-head attention |
| `[PRIMARY_METRIC]` | Headline evaluation metric | RMSE |
| `[SECONDARY_METRICS]` | Supporting metrics | MAE, R², MAPE |
| `[CLASSICAL_METHODS]` | Traditional approaches in the domain | ARIMA, kriging, dispersion models |
| `[GRAPH_CONSTRUCTION]` | How the graph is built | distance-based adjacency / correlation matrix |
| `[TEMPORAL_MODEL]` | Core sequence model type | LSTM / GRU / Transformer encoder |
| `[SPATIAL_MODEL]` | Core graph model type | GCN / GAT / GraphSAGE |

---

## Phase 1 — Intake

Collect all of the following before writing. Ask for anything missing:

| Field | Description |
|---|---|
| **Title** | Full thesis title |
| **Research problem** | The gap or challenge the work addresses |
| **Dataset(s)** | Name, source, size, spatial/temporal resolution |
| **Proposed model(s)** | Architecture names and key components |
| **Baseline(s)** | What is being compared against |
| **Experimental setup** | Train/val/test splits, metrics, hardware, hyperparameters |
| **Quantitative results** | All metric tables |
| **Ablation / qualitative findings** | Component-removal results, any visualizations described |
| **Contributions** | Bullet list of novel claims |
| **Supervisor name** | For the cover page |
| **Future directions** | At least 3–5 ideas |
| **Papers xlsx** | Optional — enables automatic literature review and bibliography |

If the user says "just write it with what you have", proceed and mark missing fields `[TBD]`.

---

## Papers Spreadsheet — Literature Review & Bibliography Builder

When the user provides a `.xlsx` file of research papers, it becomes the **sole source of
truth** for Section 2.6 (Literature Review and Research Gap) and the References list.
**Never invent or hallucinate any paper not present in this file.**

### Step 1 — Read the xlsx

Use the `xlsx` skill to open and parse the file. Expected columns:

| Column | Description |
|---|---|
| `authors` | Full author list, e.g. "Smith, J., Lee, K." |
| `year` | Publication year |
| `title` | Full paper title |
| `venue` | Journal or conference name |
| `method` / `model` | Core technique proposed |
| `dataset` | Dataset(s) used |
| `metrics` | Reported metric names and values |
| `category` | Classification — see Step 3 |
| `notes` / `summary` | Optional free-text notes |

If column names differ, infer from context or ask once before proceeding.

### Step 2 — Build the Citation Index

Assign numbers `[1]`, `[2]`, … in order of first appearance in the thesis text:

```
[1] = {authors: "Author et al.", year: 20XX, title: "...", venue: "...", key: "author20XX"}
[2] = ...
```

### Step 3 — Group Papers by Category

Map each row's `category` value to a literature review subsection.
Subsection titles use `[DOMAIN]` — adapt them to the actual research field.

| Category value | Subsection |
|---|---|
| `Traditional` | 2.6.1 Traditional / Statistical Methods |
| `Hybrid` | 2.6.2 Hybrid Domain + Deep Learning Models |
| `DeepSequence` | 2.6.3 Pure Deep Learning Sequence Models |
| `GNN` | 2.6.4 Graph Neural Networks and Attention Integration |
| `Spatiotemporal` | 2.6.5 Spatiotemporal Models with [DOMAIN]-Specific Data |
| `Other` / uncategorized | Best-fit subsection based on `method` field |

Omit any subsection that has zero papers and renumber accordingly.
If the user uses custom category names in their xlsx, use those instead.

### Step 4 — Write Section 2.6 from the Papers

Every claim about a prior work must be backed by an inline `[N]` citation.
Never write a sentence about a paper without citing it.

**Per-paper writing template** (apply inside every subsection):
1. **Opening**: "[Authors] [YEAR] [verb: proposed / introduced / evaluated / demonstrated] [method name] for [task] [N]."
2. **What it does**: 1–2 sentences on the approach.
3. **Key result**: 1 sentence. If metric values are in the xlsx: "Achieving [metric] of [value] on [dataset], the model outperformed [baseline] by [margin]."
4. **Limitation**: 1 sentence on what the method cannot capture.

**Grouping rule**: if 2+ papers share a limitation, consolidate:
"Both [A] [N] and [B] [M] rely on [shared limitation], which restricts applicability when [condition]."

**Subsection closing transitions** (adapt `[DOMAIN]` term):
- After 2.6.1: "These limitations motivate the exploration of more data-driven approaches, reviewed next."
- After 2.6.2: "Their dependence on [domain-specific structure] limits adaptability, motivating purely data-driven alternatives."
- After 2.6.3: "Despite strong temporal performance, these models [spatial limitation], an essential factor in [DOMAIN] forecasting."
- After 2.6.4: "This demonstrates the potential of combining attention with graph-based reasoning, though [remaining gap]."
- After 2.6.5: "These methods are often constrained by [recurrent bottleneck / static graph / data assumption]."

### Step 5 — Write the Research Gap Subsection (2.6.X)

Always present, regardless of xlsx size. Structure:

**Para 1 — The joint gap**: "While [MODEL_FAMILY_A] and [MODEL_FAMILY_B] have individually shown strong performance in [DOMAIN] forecasting, limited work has explored their joint integration in a unified framework."

**Para 2 — Failure mode A** (temporal-only models): treat each [SPATIAL_UNIT] independently, no spatial reasoning. One sentence per limitation.

**Para 3 — Failure mode B** (recurrent spatial models): bullet list of 3 limitations:
- Poor scalability due to sequential computation
- Difficulty capturing long-range temporal dependencies
- Limited flexibility for [domain-specific cross-node interaction]

**Para 4 — Failure mode C** (attention-only models): powerful temporally but ignore spatial structure unless explicitly modified.

**Para 5 — Gap statement**: "Despite the proven benefits of each approach, there is a notable lack of models that:" → 3 bullet points drawn directly from what is missing across the xlsx papers.

**Para 6 — Proposed solution**: "To address this gap, we propose [N] novel [DOMAIN] forecasting architectures:" → numbered list of [VARIANT_1] and [VARIANT_2] with one-sentence descriptions. Close with a shared-components bullet list.

### Step 6 — Build the References Section

From the citation index, format each entry as:
```
[N] Firstname Lastname, Firstname Lastname. Title in sentence case. Venue,
    Volume(Issue):pages, Year.
```
For arXiv entries: `arXiv preprint arXiv:XXXX.XXXXX, Year.`

Default ordering: by order of first appearance in text.
Never add a paper not in the xlsx. Never cite a paper not in References.

### Step 7 — Citation Consistency Check

Before finalizing, scan all `[N]` tags and report:
```
⚠️  Citation warnings:
- [N] cited in text but not found in xlsx
- [N] in References but never cited in body
```

---

## Phase 2 — Document Blueprint

Generate the thesis in this fixed order. Do not skip, reorder, or merge sections.
Replace every `[PLACEHOLDER]` with the user's actual research content derived from
the Terminology Map.

---

### Cover Page

```
[University Name]
[Faculty Name]

[THESIS TITLE]

Bachelor Thesis

Author:           [Full Name]
Supervisors:      [Supervisor Name]
Submission Date:  [Month, Year]
```

---

### Declaration Page

Single short paragraph: the thesis is original work; due acknowledgement has been given
to all other material used. Signed with name and date.

---

### Acknowledgments

Three to four paragraphs:
1. Opening acknowledgment (religious, philosophical, or motivational — adapt to author)
2. Personal support — family, partner, warm and specific
3. Academic support — supervisor(s) named with specific contributions
4. Optional closing quote (italicized, attributed)

Tone: warm, personal, grateful. First person. Not formal.

---

### Abstract (~350–450 words)

**Para 1 — Motivation & Stakes**: Real-world importance of [DOMAIN]. Concrete scale or
impact. Core gap: existing tools insufficient because [reason].

**Para 2 — Prior Work & Limits**: One representative prior approach ([Author et al., N]).
What it does well. Its architectural limitation this thesis addresses.

**Para 3 — Proposed Approach**: "In this thesis, we propose and evaluate [MODEL_FAMILY]…"
Key novelties: [SPATIAL_MODEL] for spatial encoding, [TEMPORAL_MODEL] for temporal modeling,
attention variant, positional encoding strategy. Mention dataset(s).

**Para 4 — Results Summary**: Best-performing model. One headline metric improvement.
One secondary finding.

**Para 5 — Future Directions**: 1–2 sentences naming 3–5 directions.

---

### Table of Contents

All chapters and numbered sections with page numbers.

---

### Chapter 1: Introduction (~1,500–2,000 words)

#### 1.1 Motivation
Three paragraphs:
- Para 1: Real-world importance of [DOMAIN]. Why accurate [TARGET_VARIABLE] prediction matters.
- Para 2: Why [CLASSICAL_METHODS] fall short — name the category of limitation.
- Para 3: Why [MODEL_FAMILY] opens a new opportunity. End: "This thesis builds upon these advances to develop robust [DOMAIN] forecasting frameworks."

#### 1.2 Related Work and Promising Approaches
One to two paragraphs. Cite 1–3 representative works from the xlsx by author+year.
Identify the gap: "Despite these innovations, a gap remains in jointly modeling [spatial structure] and [temporal dynamics] using [MODEL_FAMILY]."
End: "This thesis aims to bridge this gap by integrating [SPATIAL_MODEL] with [TEMPORAL_MODEL]."

#### 1.3 Problem Statement and Objective
One framing paragraph. Then a bullet list of 3–4 specific challenges:
- Modeling complex non-linear [DOMAIN] dynamics
- Capturing dependencies across [SPATIAL_UNITS]
- [Domain-specific challenge]
- Ensuring scalability to large [SPATIAL_COLLECTION] and noisy datasets

End: "The objective is to evaluate and improve spatiotemporal deep learning models by [approach]. We propose [VARIANT_1] and [VARIANT_2] and test them on [DATASET_NAME]."

#### 1.4 Contributions
Bullet list of 4–6, each starting with an action noun:
- Proposal of [N] novel [DOMAIN] forecasting architectures…
- Comparative analysis against [BASELINE_MODEL] across [conditions]…
- Demonstration of [key finding]…
- Extensive preprocessing pipeline for [DATASET_NAME]…
- Open-source codebase at [github.com/username/repo]…

#### 1.5 Thesis Outline
One sentence per chapter.

---

### Chapter 2: Background (~3,500–5,000 words)

Opening paragraph maps all sections by number.

#### 2.1 Traditional Approaches to [DOMAIN] Forecasting
Two to three pages. For 2–3 classical methods from [CLASSICAL_METHODS]:
- Definition or intuition
- Strengths
- Key limitations (each as its own sentence or bullet)

Closing: "To address these limitations, researchers have turned to deep learning, which we review next."

#### 2.2 Deep Sequence Models for [DOMAIN] Forecasting
Two to three pages. Cover the progression relevant to the work (e.g., RNN → LSTM → GRU).
For each model:
- Name and formal description
- Figure reference: "Figure 2.X: Architecture of [model]."
- Application in [DOMAIN] — cite 1 paper from xlsx
- Limitation motivating the next subsection

Closing: bridge to 2.3.

#### 2.3 [TEMPORAL_MODEL] for Temporal Modeling
Two pages. Subsections:
- Architecture overview (figure reference)
- Self-attention and multi-head attention — include formula:
  `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V`
- Masking strategy used in this work (causal or bidirectional — state which and why)
- Positional encoding choice (learnable vs. sinusoidal — state justification)
- Feed-forward network: `FFN(x) = max(0, xW_1 + b_1)W_2 + b_2`
- Dropout and regularization

#### 2.4 Graph Neural Networks for Spatial [DOMAIN] Modeling
One to two pages. Subsections:
- Graph theory foundations: G = (V, E), adjacency matrix A, degree matrix D
- Spectral graph convolution: normalized Laplacian, convolution formula
- Node representation: what [SPATIAL_UNITS] represent as nodes, what [EDGE_TYPE] represents as edges
- Strengths of GNNs in [DOMAIN] (3–4 bullets)

#### 2.5 Dataset(s)
One page. For each dataset:
- Source and provenance
- Spatial granularity: [SPATIAL_UNITS], coverage area
- Temporal granularity and time span
- Size: N nodes, E edges, T time steps
- Graph construction: [GRAPH_CONSTRUCTION]
- Normalization approach

#### 2.6 Literature Review and Research Gap
Content driven entirely by the Papers Spreadsheet pipeline above.
Subsections: 2.6.1 → 2.6.2 → 2.6.3 → 2.6.4 → 2.6.5 → 2.6.X Research Gap.
Subsections with zero papers are omitted and remaining ones renumbered.

---

### Chapter 3: Methodology (~3,500–5,000 words)

Opening: "This chapter outlines the methodological framework used to develop, train, and evaluate the proposed [DOMAIN] forecasting models."

#### 3.1 Problem Definition
- Variable list as bullets: N ([SPATIAL_UNITS]), T (input window), F (features per [SPATIAL_UNIT]), H (forecast horizon)
- Input tensor: `X ∈ R^{B×T×N×F}`
- Output tensor: `Ŷ ∈ R^{B×N×H}`
- Additional inputs: graph G, edge weights w_ij, static node features Z
- Mapping: `F : (X, G, Z) → Ŷ`
- **Forecasting Setup**: justify T and H choices, reference baseline paper
- **Dataset Context**: one sentence per dataset on preprocessing outcome

#### 3.2 Model Architecture
Opening paragraph summarizing all models evaluated.

**3.2.1 [BASELINE_MODEL]**
- Architecture description
- Core computational equation
- Input/output shapes
- Key limitation

**3.2.2 [VARIANT_1]**
- What replaces the baseline component and why
- Temporal modeling scope
- Spatial encoding approach
- Remaining limitation

**3.2.3 [VARIANT_2] — Full Proposed Model**
Sub-steps:
1. Input projection → hidden dimension D
2. Spatial encoding: `S = [SPATIAL_MODEL](Z, G) ∈ R^{N×D}` → broadcast across T
3. Positional encoding: learnable `P ∈ R^{T×N×D}` added
4. `X_enc = X_proj + S_seq + P` → flatten to [B, T·N, D] → [TEMPORAL_MODEL] with causal mask
5. Reshape → take H_T → MLP → `Ŷ`

Key Properties (3 bullets):
- [Global/local] attention across [SPATIAL_UNITS] and time
- [SPATIAL_MODEL] captures [EDGE_TYPE] spatial structure
- Learnable temporal embeddings for flexible time alignment

**3.2.4 Architectural Comparison Table**

| Model | Temporal Modeling | Spatial Modeling | Cross-node Attention |
|---|---|---|---|
| [BASELINE_MODEL] | [method] | [method] | ✗ |
| [VARIANT_1] | [method] | [method] | ✗ |
| [VARIANT_2] | [method] | [method] | ✓ |

#### 3.3 Data Preprocessing
Opening: "Accurate [DOMAIN] forecasting requires not only a strong model architecture but also clean, well-structured input data."

For each dataset:
- *[SPATIAL_UNIT] Selection and Filtering*: quality criteria as bullets with thresholds
- *[TARGET_VARIABLE] Normalization*: z-score formula; any domain-specific normalization
- *Graph Construction*: [GRAPH_CONSTRUCTION] method, edge weight formula, sparsification strategy

Closing **Summary**: 3 bullets confirming consistency.

#### 3.4 Training Setup

**3.4.1 Sliding Window Strategy**
Input window T, output horizon H, feature used, sample multiplication benefit, input shape.

**3.4.2 Loss Function and Optimizer**
Loss formula, Adam optimizer, initial LR, StepLR scheduler (decay rate and interval per dataset).

**3.4.3 Hardware and Device Setup**
Hardware, GPU/CPU fallback, set_seed reproducibility snippet.

#### 3.5 Evaluation Metrics
One subsection per metric — formula, rationale, granularities reported:
- **[PRIMARY_METRIC]**: magnitude accuracy
- **[SECONDARY_METRIC_1]**: relative / scale-invariant error
- **[SECONDARY_METRIC_2]**: directional accuracy or variance explained
- **Reconstruction**: inverse normalization `ŷ_orig = ẑ·σ + μ`
- **Evaluation Protocol**: held-out test set, what is reported

#### 3.6 Implementation Details
Software stack, hardware, training config table, reproducibility snippet, GitHub link.

#### 3.7 Limitations and Assumptions
- Data quality issues per dataset
- Graph assumptions (static, backbone trade-offs)
- Modeling constraints (fixed horizon, normalization stationarity, positional encoding limits)
- Evaluation assumptions (non-stationarity, metric limitations)
- Generalization scope

---

### Chapter 4: Results and Discussion (~5,000–7,000 words)

#### 4.1 Overview of Experimental Setup
- Goals: 4-bullet list
- Metrics: Metric: description format
- Per-dataset experiment overview
- Complexity note if applicable (time/memory formulas, mitigation strategy, code snippet)

#### 4.2 Experiments

"We conducted a total of N experiments: X for [DATASET_1], Y for [DATASET_2]."

For every experiment, repeat:
```
**Experiment N — [MODEL] ([DATA_SUBSET])**
[One-sentence configuration description]
**Purpose:** [What this isolates]
[Figure N.M: [MODEL] on [SPATIAL_UNIT_NAME] ([DATASET])]
**Results:** [2–3 sentences: what improved, what didn't, why]
[[SPATIAL_UNIT]-level metrics | Average across all [SPATIAL_UNITS]]
[Comparison table — Metric | Model A | Model B | (%) change]
```

#### 4.3 Results on [DATASET_1]

**4.3.1 Quantitative Results**
Master table: all models × all subsets × all metrics. Bold best per row.

**4.3.2 Discussion**
- *Impact of Data Quality*
- *Model Behavior and Design Tradeoffs*: one paragraph per model
- *Scale Effects on [PRIMARY_METRIC]*: why secondary metrics are needed
- *Summary of Findings*: 5 bullets

#### 4.4 Results on [DATASET_2] (if applicable)

**4.4.1 Quantitative Results**
Table + figure reference.

**4.4.2 Discussion**
- *Effect of Dataset Characteristics*
- *[BASELINE_MODEL]: Reasonable but Limited*
- *[VARIANT_1]: Best All-Around Performer* (with % improvements)
- *[VARIANT_2]: Superior [strength] but [trade-off]*
- *Interesting Edge Case*: where the "worse" model won; mechanism; 3 future fixes
- *Summary of Key Insights*: 6 bullets

#### 4.5 Ablation and Qualitative Insights

1. **Baseline Enhancement**: hyperparameters tuned, before/after metric numbers
2. **Proposed Model Scaling**: hidden size, heads, epochs — before/after
3. **Spatial Encoder Ablation**: two-column table (Without [SPATIAL_MODEL] | With [SPATIAL_MODEL]),
   interpretation paragraph per metric, closing sentence on hybrid value

**Takeaways**: 5 bullets with bold lead phrases.

#### 4.6 Overall Discussion and Takeaways
8 bullets: `• **Bold claim.** 2–3 sentences of evidence.`
1. Best model + computational constraint
2. Best scalability/accuracy tradeoff
3. Attention models' superior trend detection
4. Practical deployment potential
5. Baseline failure mode
6. Preprocessing importance
7. [SPATIAL_MODEL] encoder calibration contribution
8. Metric diversity necessity

#### 4.7 Summary
~120 words. Best tradeoff model, richest-reasoning model, headline result, foundation for
next-generation [DOMAIN] tools.

---

### Chapter 5: Conclusion & Future Work (~1,000–1,500 words)

#### 5.1 Conclusion
- Para 1: "This thesis set out to evaluate…"
- Para 2: Strongest overall model
- Para 3: [PRIMARY_METRIC] / trend detection contribution
- Para 4: Preprocessing / [SPATIAL_MODEL] encoder role
- Para 5: "From a broader perspective, this thesis contributes…"
- Para 6: "In conclusion, this work demonstrates that…" + forward-looking sentence

#### 5.2 Future Work
5 numbered bullets — **Bold title** + 2–3 sentences citing 1–2 example approaches:
1. Hybrid architectures (attention + recurrence)
2. Sparse / linear attention for scalability
3. Transfer learning and generalization
4. Dynamic graph construction
5. Multi-modal / multi-source data fusion

---

### References

**xlsx provided**: generate entirely from citation index. No invented entries.
**No xlsx**: `[Author et al., YEAR — details TBD]` placeholders only.

Format:
```
[N] Firstname Lastname, Firstname Lastname. Title in sentence case.
    Venue, Volume(Issue):pages, Year.
```
arXiv: `arXiv preprint arXiv:XXXX.XXXXX, Year.`

---

### Appendix (if applicable)

Extra figures, extended tables, or implementation details.

---

## Style Guide

**Tone**: formal academic English; no contractions; active voice for contributions;
passive acceptable for methodology; hedge interpretations ("likely due to", "suggests that").

**Paragraphs**: 3–5 sentences; topic sentence first; evidence in middle; implication last.

**Figures**: `Figure N.M: [Title with model name and dataset/[SPATIAL_UNIT]].` Always
reference before the figure appears.

**Tables**: bold best per row. Caption below: `Table N.M: [Description]. Bolded values
indicate best performance per row.`

**Equations**: number only when referenced later; introduce all variables immediately after;
LaTeX-style notation `X ∈ R^{T×N×F}`.

**Transitions**: end each section with a bridge sentence pointing to the next.

**Metrics**: percentage changes always signed (`−32.85%`); 2 decimal places; always state
improvement direction explicitly.

---

## Quick Reference: Word Counts

| Section | Target |
|---|---|
| Abstract | ~400 words |
| Ch 1 Introduction | ~1,500–2,000 words |
| Ch 2 Background | ~3,500–5,000 words |
| Ch 3 Methodology | ~3,500–5,000 words |
| Ch 4 Results | ~5,000–7,000 words |
| Ch 5 Conclusion | ~1,000–1,500 words |
| **Total** | **~15,000–21,000 words** |

---

## Output Format

- **Claude Code**: save to `./thesis_[short_title].md`
- **claude.ai**: save to `/mnt/user-data/outputs/thesis_[short_title].md`
- Word doc requested: additionally run the `docx` skill on the markdown output
- Individual chapters requested: one file per chapter

Confirm before writing:
> "Terminology map confirmed. Ready to generate the full thesis (~15,000–21,000 words). Shall I proceed?"

---

## Behavior Rules

- **Terminology map first** — resolve all `[PLACEHOLDER]`s before writing any prose
- **Missing inputs** → ask; never invent results
- **Incomplete results** → mark `[TBD]`, note at top of output
- **xlsx provided** → Section 2.6 and References come entirely from it; zero invented citations
- **No xlsx** → `[Author et al., YEAR]` placeholders only; never invent titles or authors
- **Citation discipline** → every prior-work claim gets `[N]`; no exceptions
- **No domain lock-in** → never hardcode epidemic/COVID/city/mobility or any other domain terms; use the Terminology Map throughout
- **No copying** → all prose is original; the blueprint governs structure, not wording
- **Consistency** → figures, tables, section numbers, and citation numbers must match throughout
- **Depth** → never outline where the blueprint calls for full prose
