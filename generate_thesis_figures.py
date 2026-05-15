"""
generate_thesis_figures.py
Generates all thesis figures for:
  GUC Bachelor Thesis — Spatio-Temporal PM2.5 Forecasting
  Beijing Multi-Site Air Quality Dataset, 12 Stations

Run:  python generate_thesis_figures.py
Output: Figures/thesis/*.pdf  (and matching .png for quick preview)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# ─── Output directory ────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), 'Figures', 'thesis')
os.makedirs(OUT, exist_ok=True)

# ─── Global style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'primary':   '#2C6FAC',
    'secondary': '#E07B39',
    'accent':    '#3DAA6A',
    'neutral':   '#7B7B7B',
    'light':     '#D9E8F5',
    'highlight': '#F5C842',
    'red':       '#C0392B',
    'purple':    '#7D3C98',
    'teal':      '#1A7A7A',
}

def save(fig, name):
    path_pdf = os.path.join(OUT, name + '.pdf')
    path_png = os.path.join(OUT, name + '.png')
    fig.savefig(path_pdf)
    fig.savefig(path_png)
    plt.close(fig)
    print(f'  OK {name}.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 3 — DATA & SETUP FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def fig_station_map():
    """Beijing monitoring stations scatter map."""
    stations = {
        'Aotizhongxin':   (116.397, 39.982),
        'Changping':      (116.230, 40.220),
        'Dingling':       (116.220, 40.292),
        'Dongsi':         (116.417, 39.929),
        'Guanyuan':       (116.339, 39.929),
        'Gucheng':        (116.184, 39.914),
        'Huairou':        (116.628, 40.328),
        'Nongzhanguan':   (116.461, 39.937),
        'Shunyi':         (116.655, 40.128),
        'Tiantan':        (116.407, 39.886),
        'Wanliu':         (116.287, 39.987),
        'Wanshouxigong':  (116.352, 39.878),
    }
    lons = [v[0] for v in stations.values()]
    lats = [v[1] for v in stations.values()]
    names = list(stations.keys())

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(lons, lats, s=120, color=COLORS['primary'], zorder=5, edgecolors='white', linewidth=1.0)

    offsets = {
        'Aotizhongxin':   (0.01, 0.007),
        'Changping':      (-0.09, 0.007),
        'Dingling':       (0.012, 0.007),
        'Dongsi':         (0.012, 0.007),
        'Guanyuan':       (-0.085, -0.012),
        'Gucheng':        (-0.075, 0.007),
        'Huairou':        (0.012, 0.007),
        'Nongzhanguan':   (0.012, -0.014),
        'Shunyi':         (0.012, 0.007),
        'Tiantan':        (0.012, -0.013),
        'Wanliu':         (-0.065, 0.007),
        'Wanshouxigong':  (0.012, -0.013),
    }
    for name, (lon, lat) in stations.items():
        dx, dy = offsets[name]
        ax.annotate(name, (lon, lat), xytext=(lon + dx, lat + dy),
                    fontsize=8.5, ha='left')

    # Draw rough boundary box for Beijing
    ax.set_xlim(116.05, 116.82)
    ax.set_ylim(39.79, 40.43)
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title('Beijing Air Quality Monitoring Network (12 Stations)')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_aspect('equal')

    # Add north arrow
    ax.annotate('N', xy=(0.96, 0.93), xycoords='axes fraction',
                fontsize=13, ha='center', va='center', fontweight='bold')
    ax.annotate('', xy=(0.96, 0.97), xytext=(0.96, 0.93), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    save(fig, 'fig_station_map')


def fig_pm25_timeseries():
    """PM2.5 overview: monthly mean + raw excerpt from two contrasting stations."""
    raw_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
    station_files = {
        'Dongsi':     'PRSA_Data_Dongsi_20130301-20170228.csv',
        'Dingling':   'PRSA_Data_Dingling_20130301-20170228.csv',
    }
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    # Top: full time series monthly mean for all 12 stations
    all_monthly = []
    all_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    for fname in sorted(all_files):
        df = pd.read_csv(os.path.join(raw_dir, fname))
        # Normalise column names — some files may use 'date' instead of 'datetime'
        col_map = {c: c.lower().strip() for c in df.columns}
        df = df.rename(columns=col_map)
        date_col = next((c for c in df.columns if 'date' in c or 'time' in c), None)
        if date_col is None:
            continue
        df[date_col] = pd.to_datetime(df[date_col])
        pm_col = next((c for c in df.columns if 'pm2' in c.lower()), None)
        if pm_col is None:
            continue
        series = df.set_index(date_col)[pm_col].dropna()
        monthly = series.resample('ME').mean()
        all_monthly.append(monthly)

    combined = pd.concat(all_monthly, axis=1)
    mean_monthly = combined.mean(axis=1)
    std_monthly  = combined.std(axis=1)

    ax = axes[0]
    ax.fill_between(mean_monthly.index, mean_monthly - std_monthly,
                    mean_monthly + std_monthly, alpha=0.2, color=COLORS['primary'], label='±1 SD across stations')
    ax.plot(mean_monthly.index, mean_monthly, color=COLORS['primary'], lw=1.5, label='Network mean')
    ax.set_ylabel('PM2.5 (µg/m³)')
    ax.set_title('Monthly Mean PM2.5 — All 12 Stations (2013–2017)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(bottom=0)

    # Bottom: 7-day window raw hourly excerpt comparing urban vs suburban
    ax2 = axes[1]
    colors_map = {'Dongsi': COLORS['primary'], 'Dingling': COLORS['secondary']}
    for sname, fname in station_files.items():
        df = pd.read_csv(os.path.join(raw_dir, fname))
        col_map = {c: c.lower().strip() for c in df.columns}
        df = df.rename(columns=col_map)
        date_col = next((c for c in df.columns if 'date' in c or 'time' in c), None)
        df[date_col] = pd.to_datetime(df[date_col])
        pm_col = next((c for c in df.columns if 'pm2' in c.lower()), None)
        df = df.set_index(date_col)[pm_col].dropna()
        # Pick a representative winter week (high pollution period)
        window = df.loc['2016-01-01':'2016-01-14']
        ax2.plot(window.index, window.values, lw=1.2, label=sname, color=colors_map[sname], alpha=0.85)

    ax2.axhline(150, color=COLORS['red'], lw=0.8, linestyle='--', alpha=0.7, label='WHO 24h guideline (15 µg/m³)')
    ax2.set_ylabel('PM2.5 (µg/m³)')
    ax2.set_xlabel('Date')
    ax2.set_title('Hourly PM2.5 — Urban (Dongsi) vs. Suburban (Dingling), Jan 2016')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    save(fig, 'fig_pm25_timeseries')


# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 2 — BACKGROUND DIAGRAMS
# ═══════════════════════════════════════════════════════════════════════════════

def draw_box(ax, x, y, w, h, label, facecolor, edgecolor='#333333', fontsize=9, radius=0.03):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f'round,pad=0.01,rounding_size={radius}',
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, zorder=4,
            multialignment='center')

def draw_arrow(ax, x1, y1, x2, y2, color='#444444'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))


def fig_lstm_cell():
    """LSTM cell unrolled one step — gates diagram."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
    ax.set_title('Long Short-Term Memory (LSTM) Cell', fontsize=13, pad=10)

    gate_colors = {
        'Forget\nGate σ':  '#FADBD8',
        'Input\nGate σ':   '#D5E8D4',
        'Cell\nUpdate tanh': '#DAE8FC',
        'Output\nGate σ':  '#FFF2CC',
    }
    gate_x = [1.8, 3.8, 5.8, 7.8]
    gate_y = 2.5
    gw, gh = 1.3, 0.9

    for (label, color), gx in zip(gate_colors.items(), gate_x):
        draw_box(ax, gx, gate_y, gw, gh, label, facecolor=color, fontsize=8.5)

    # Cell state line (top)
    ax.annotate('', xy=(9.2, 4.2), xytext=(0.8, 4.2),
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2.0))
    ax.text(5.0, 4.5, '$C_{t-1}$ → cell state → $C_t$',
            ha='center', fontsize=9.5, color=COLORS['primary'], style='italic')

    # h_{t-1} input
    ax.annotate('', xy=(0.8, 2.5), xytext=(0.1, 2.5),
                arrowprops=dict(arrowstyle='->', color='#444', lw=1.4))
    ax.text(0.45, 2.8, '$h_{t-1}$', fontsize=9, ha='center')

    # x_t input
    ax.annotate('', xy=(0.8, 1.4), xytext=(0.1, 1.4),
                arrowprops=dict(arrowstyle='->', color='#444', lw=1.4))
    ax.text(0.45, 1.1, '$x_t$', fontsize=9, ha='center')

    # Output h_t
    ax.annotate('', xy=(9.8, 2.5), xytext=(9.2, 2.5),
                arrowprops=dict(arrowstyle='->', color='#444', lw=1.4))
    ax.text(9.55, 2.8, '$h_t$', fontsize=9, ha='center')

    # Vertical connections (gate → cell state)
    for gx in gate_x:
        ax.plot([gx, gx], [gate_y + gh/2, 4.2 - 0.05], color='#888', lw=1.0, linestyle='--')

    # Multiply/add symbols on cell state line
    ops = [(2.8, '×'), (4.8, '+'), (6.8, '×')]
    for ox, sym in ops:
        circ = plt.Circle((ox, 4.2), 0.22, color='white', ec='#444', lw=1.2, zorder=5)
        ax.add_patch(circ)
        ax.text(ox, 4.2, sym, ha='center', va='center', fontsize=12, zorder=6)

    save(fig, 'fig_lstm_cell')


def fig_transformer_encoder():
    """Transformer encoder block diagram."""
    fig, ax = plt.subplots(figsize=(5, 8))
    ax.set_xlim(0, 5); ax.set_ylim(0, 9); ax.axis('off')
    ax.set_title('Transformer Encoder Block', fontsize=13, pad=8)

    blocks = [
        (2.5, 1.0,  2.8, 0.65, 'Input\n$X \\in \\mathbb{R}^{T \\times N \\times D}$',  '#F5F5F5'),
        (2.5, 2.1,  2.8, 0.65, 'Positional Encoding\n(Learnable)',                      '#E8F4FD'),
        (2.5, 3.3,  2.8, 0.75, 'Multi-Head\nSelf-Attention',                            COLORS['light']),
        (2.5, 4.35, 2.8, 0.55, 'Add & Layer Norm',                                      '#F0F0F0'),
        (2.5, 5.3,  2.8, 0.75, 'Feed-Forward Network\n(SegMoE / Standard FFN)',         '#E8F8F0'),
        (2.5, 6.35, 2.8, 0.55, 'Add & Layer Norm',                                      '#F0F0F0'),
        (2.5, 7.4,  2.8, 0.65, 'Encoder Output\n$Z \\in \\mathbb{R}^{T \\times N \\times D}$', '#F5F5F5'),
    ]
    for (x, y, w, h, label, fc) in blocks:
        draw_box(ax, x, y, w, h, label, facecolor=fc, fontsize=9)

    # Arrows between blocks
    ys = [b[1] for b in blocks]
    for i in range(len(ys)-1):
        y_bottom = ys[i] + blocks[i][3]/2
        y_top    = ys[i+1] - blocks[i+1][3]/2
        draw_arrow(ax, 2.5, y_bottom, 2.5, y_top)

    # Residual skip arrows
    for skip_start, skip_end in [(2.1, 4.35), (4.65, 6.35)]:
        ax.annotate('', xy=(3.9 + 0.05, skip_end),
                    xytext=(3.9 + 0.05, skip_start),
                    arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=1.3,
                                   connectionstyle='arc3,rad=0.0'))
        ax.text(4.25, (skip_start + skip_end)/2, 'residual', fontsize=7.5,
                color=COLORS['secondary'], rotation=90, va='center')

    # Repeat brace
    ax.text(0.3, 4.85, '×$N_L$\nlayers', fontsize=9, ha='center', va='center',
            color='#555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FAFAFA', edgecolor='#AAA'))

    save(fig, 'fig_transformer_encoder')


def fig_gat_attention():
    """GAT multi-head attention on a 5-node subgraph."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(-0.5, 7); ax.set_ylim(-0.5, 5); ax.axis('off')
    ax.set_title('Graph Attention Network (GAT) — Message Passing Step', fontsize=12, pad=8)

    # Node positions
    center = (2.0, 2.3)
    neighbors = [(0.2, 3.8), (0.3, 0.8), (2.0, 4.3), (3.8, 3.8), (3.7, 0.8)]
    labels   = ['$v_1$', '$v_2$', '$v_3$', '$v_4$', '$v_5$']

    def draw_node(ax, pos, label, central=False):
        c = COLORS['primary'] if central else COLORS['secondary']
        r = 0.38 if central else 0.30
        circle = plt.Circle(pos, r, color=c, zorder=4, ec='white', lw=1.5)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=10,
                color='white', fontweight='bold', zorder=5)

    # Edges with attention weight labels
    alpha_vals = [0.35, 0.18, 0.22, 0.14, 0.11]
    for nb, lbl, alpha in zip(neighbors, labels, alpha_vals):
        lw = 0.8 + alpha * 5
        ax.plot([center[0], nb[0]], [center[1], nb[1]],
                color='#888', lw=lw, alpha=0.6, zorder=2)
        mx = (center[0] + nb[0]) / 2 + 0.15
        my = (center[1] + nb[1]) / 2
        ax.text(mx, my, f'$\\alpha={alpha:.2f}$', fontsize=7.5, color='#444',
                ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=0.5))

    for nb, lbl in zip(neighbors, labels):
        draw_node(ax, nb, lbl)
    draw_node(ax, center, '$v_0$', central=True)

    # Formula on the right
    ax.text(5.0, 3.8, 'Attention weight:', fontsize=9.5, ha='left', color='#222')
    ax.text(5.0, 3.2,
            r'$\alpha_{ij} = \frac{\exp(\mathrm{LeakyReLU}(a^T[Wh_i \| Wh_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(\cdot)}$',
            fontsize=8.5, ha='left', color='#222')
    ax.text(5.0, 2.1, 'Aggregation:', fontsize=9.5, ha='left', color='#222')
    ax.text(5.0, 1.5,
            r"$h'_i = \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j\right)$",
            fontsize=9.5, ha='left', color='#222')

    save(fig, 'fig_gat_attention')


# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 3 — MODEL ARCHITECTURE DIAGRAMS
# ═══════════════════════════════════════════════════════════════════════════════

def fig_model_architecture():
    """Full GraphTransformer (temporal-first + SegMoE) pipeline."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 11); ax.set_ylim(0, 9); ax.axis('off')
    ax.set_title('Proposed Model: Temporal-First GraphTransformer with SegMoE FFN', fontsize=12, pad=10)

    # Main pipeline blocks (left-to-right flow)
    pipeline = [
        (1.2, 4.5, 1.7, 0.7,  'Input\n$X \\in \\mathbb{R}^{B \\times T \\times N \\times 33}$', '#F5F5F5'),
        (3.1, 4.5, 1.8, 0.7,  'Linear Input\nProjection\n$33 \\rightarrow D=64$',                '#E8F4FD'),
        (5.2, 6.3, 2.2, 1.2,  'Transformer Encoder\n($N_L=2$ layers)\n+SegMoE FFN\n+Node Embeddings', COLORS['light']),
        (5.2, 4.5, 2.2, 0.75, 'Dynamic Wind-Aware\nAdjacency $A_t$',                              '#FEF9E7'),
        (5.2, 2.7, 2.2, 1.1,  'GATv1 Spatial\n($K=1$ layer)\n+Learnable $\\alpha$ gate',          '#E8F8F0'),
        (8.3, 4.5, 1.9, 0.75, 'Persistence\nResidual\n$\\hat{y}_{last}$',                         '#FDEDEC'),
        (9.8, 4.5, 1.3, 0.7,  'Output\n$\\hat{Y} \\in \\mathbb{R}^{B \\times 6 \\times N}$',       '#F5F5F5'),
    ]
    for (x, y, w, h, label, fc) in pipeline:
        draw_box(ax, x, y, w, h, label, facecolor=fc, fontsize=8.5)

    # Station x Horizon Bias (small box below output)
    draw_box(ax, 9.1, 3.3, 1.5, 0.55, 'Station×Horizon\nBias ($6×12$)', facecolor='#F0E6FF', fontsize=8)

    # Arrows
    # Input → Projection
    draw_arrow(ax, 1.2 + 1.7/2, 4.5, 3.1 - 1.8/2, 4.5)
    # Projection → Transformer (up)
    draw_arrow(ax, 3.1 + 1.8/2 + 0.1, 4.8, 5.2 - 2.2/2, 6.3)
    # Transformer → GAT (down-left)
    draw_arrow(ax, 5.2, 6.3 - 1.2/2, 5.2, 2.7 + 1.1/2)
    # Projection → Adjacency (direct right)
    draw_arrow(ax, 3.1 + 1.8/2, 4.5, 5.2 - 2.2/2, 4.5)
    # Adjacency → GAT
    draw_arrow(ax, 5.2, 4.5 - 0.75/2, 5.2, 2.7 + 1.1/2 + 0.05)
    # GAT → Residual
    draw_arrow(ax, 5.2 + 2.2/2, 2.7 + 0.5, 8.3 - 1.9/2, 4.5 - 0.2)
    # Residual → Output
    draw_arrow(ax, 8.3 + 1.9/2, 4.5, 9.8 - 1.3/2, 4.5)
    # Bias → Output
    ax.annotate('', xy=(9.8 - 1.3/2, 4.5 - 0.25),
                xytext=(9.1 + 1.5/2, 3.3 + 0.55/2),
                arrowprops=dict(arrowstyle='->', color=COLORS['purple'], lw=1.1, linestyle='dashed'))

    # EVT loss label
    draw_box(ax, 7.0, 1.2, 2.4, 0.6, 'EVT Hybrid Loss\n(MSE + tail penalty)', facecolor='#FDEBD0', fontsize=8.5)
    draw_arrow(ax, 9.8, 4.5 - 0.7/2, 9.8, 1.2 + 0.6/2)
    ax.annotate('', xy=(7.0 + 2.4/2, 1.2), xytext=(9.8, 1.2),
                arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=1.1))

    # Label the temporal-first ordering
    ax.text(5.2, 7.8, '① Temporal first (Transformer runs before GAT)',
            ha='center', fontsize=9, color=COLORS['primary'], style='italic')
    ax.annotate('', xy=(5.2, 7.6), xytext=(5.2, 7.2),
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=0.8))

    save(fig, 'fig_model_architecture')


def fig_segmoe_block():
    """Seg-MoE FFN block — 2-expert soft-routing architecture."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis('off')
    ax.set_title('Seg-MoE FFN Block (Replaces Standard Transformer FFN)', fontsize=12, pad=8)

    # ── Left: Standard FFN ──────────────────────────────────────────────────
    ax.text(1.7, 6.4, 'Standard FFN', ha='center', fontsize=10, fontweight='bold', color='#444')
    std_blocks = [
        (1.7, 5.4, 2.0, 0.6, 'Input $x \\in \\mathbb{R}^D$',   '#F5F5F5'),
        (1.7, 4.2, 2.0, 0.6, 'Linear($D, 4D$) + ReLU',          '#E8F4FD'),
        (1.7, 3.0, 2.0, 0.6, 'Linear($4D, D$)',                  '#E8F4FD'),
        (1.7, 1.9, 2.0, 0.6, "Output $x' \\in \\mathbb{R}^D$",  '#F5F5F5'),
    ]
    for (x, y, w, h, lbl, fc) in std_blocks:
        draw_box(ax, x, y, w, h, lbl, facecolor=fc, fontsize=8.5)
    for i in range(len(std_blocks) - 1):
        draw_arrow(ax, std_blocks[i][0], std_blocks[i][1] - std_blocks[i][3] / 2,
                       std_blocks[i+1][0], std_blocks[i+1][1] + std_blocks[i+1][3] / 2)

    # Divider
    ax.plot([4.1, 4.1], [0.5, 6.8], color='#CCCCCC', lw=1.2, linestyle='--')
    ax.text(4.1, 0.2, 'replaced by →', ha='center', fontsize=9, color='#888')

    # ── Right: actual Seg-MoE (2-expert soft routing) ───────────────────────
    ax.text(7.1, 6.4, 'Seg-MoE FFN', ha='center', fontsize=10,
            fontweight='bold', color=COLORS['primary'])

    # Input box: center (7.1, 5.5), top=5.775, bottom=5.225
    draw_box(ax, 7.1, 5.5, 2.4, 0.55, 'Input $x \\in \\mathbb{R}^D$',
             facecolor='#F5F5F5', fontsize=8.5)

    # Fan arrows: input bottom (5.225) → expert tops (3.95+0.425=4.375)
    ax.annotate('', xy=(5.55, 4.375), xytext=(6.5, 5.225),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.1))
    ax.annotate('', xy=(8.65, 4.375), xytext=(7.7, 5.225),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.1))

    # Expert boxes: center y=3.95, h=0.85 → top=4.375, bottom=3.525
    draw_box(ax, 5.55, 3.95, 1.9, 0.85,
             'Expert FFN 1\nLinear($D,4D$)+ReLU\nLinear($4D,D$)',
             facecolor=COLORS['light'], fontsize=7.5)
    draw_box(ax, 8.65, 3.95, 1.9, 0.85,
             'Expert FFN 2\nLinear($D,4D$)+ReLU\nLinear($4D,D$)',
             facecolor='#E8F8F0', fontsize=7.5)

    # Convergence arrows: expert bottoms (3.525) → weighted sum top (2.35+0.325=2.675)
    ax.annotate('', xy=(6.35, 2.675), xytext=(5.55, 3.525),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.1))
    ax.text(5.5, 3.08, '$g_1$', fontsize=10, color=COLORS['primary'],
            ha='center', fontweight='bold')

    ax.annotate('', xy=(7.85, 2.675), xytext=(8.65, 3.525),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.1))
    ax.text(8.7, 3.08, '$g_2$', fontsize=10, color=COLORS['primary'],
            ha='center', fontweight='bold')

    # Weighted sum box: center (7.1, 2.35), h=0.65 → top=2.675, bottom=2.025
    draw_box(ax, 7.1, 2.35, 3.6, 0.65,
             'Weighted sum: $g_1 \\cdot E_1(x) + g_2 \\cdot E_2(x)$',
             facecolor='#E8F4FD', fontsize=8.2)

    # Arrow: sum bottom (2.025) → output top (1.1+0.275=1.375)
    draw_arrow(ax, 7.1, 2.025, 7.1, 1.375)

    # Output box: center (7.1, 1.1)
    draw_box(ax, 7.1, 1.1, 2.4, 0.55, "Output $x' \\in \\mathbb{R}^D$",
             facecolor='#F5F5F5', fontsize=8.5)

    # Router formula note (bottom of right panel, below output)
    ax.text(7.1, 0.5,
            r'Soft router: $\mathbf{g} = \mathrm{softmax}(\mathbf{W}_r \bar{z}_{\mathrm{PM_{2.5}}}) \in \mathbb{R}^2$'
            '   ($g_1 + g_2 = 1$)',
            ha='center', va='center', fontsize=7.5, color='#555', style='italic')

    save(fig, 'fig_segmoe_block')


def fig_wind_adjacency():
    """Dynamic wind-aware adjacency construction diagram."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
    ax.set_title('Dynamic Wind-Aware Adjacency Construction', fontsize=12, pad=8)

    steps = [
        (0.9,  3.0, 1.4, 0.8, 'Raw Inputs\nWind Speed\n$wspm$\nWind Dir $wd$', '#F5F5F5'),
        (3.0,  4.2, 1.9, 0.75, 'Distance\nAdjacency $A_{dist}$\n(Haversine decay)',       '#E8F4FD'),
        (3.0,  1.8, 1.9, 0.75, 'Wind Transport\nAdjacency $A_{wind}$\n(circular mean)',    '#E8F8F0'),
        (5.8,  3.0, 2.1, 0.8,  'Mixture:\n$(1-\\alpha)A_{dist} + \\alpha A_{wind}$\n$\\alpha$ learned', '#FEF9E7'),
        (8.5,  3.0, 1.5, 0.75, 'Final\nAdjacency\n$A_t$ per batch',                      COLORS['light']),
    ]

    for (x, y, w, h, lbl, fc) in steps:
        draw_box(ax, x, y, w, h, lbl, facecolor=fc, fontsize=8.5)

    # Raw → distance and wind
    draw_arrow(ax, 0.9+1.4/2, 3.3, 3.0-1.9/2, 4.2)
    draw_arrow(ax, 0.9+1.4/2, 2.7, 3.0-1.9/2, 1.8)

    # Distance + wind → mixture
    draw_arrow(ax, 3.0+1.9/2, 4.2, 5.8-2.1/2, 3.2)
    draw_arrow(ax, 3.0+1.9/2, 1.8, 5.8-2.1/2, 2.8)

    # Mixture → final
    draw_arrow(ax, 5.8+2.1/2, 3.0, 8.5-1.5/2, 3.0)

    # Annotations
    ax.text(2.0, 5.0, 'Static\n(batch-shared)', fontsize=8, ha='center', color='#666',
            bbox=dict(facecolor='#F8F8F8', edgecolor='#DDD', boxstyle='round,pad=0.25'))
    ax.annotate('', xy=(3.0, 4.6), xytext=(2.2, 5.0),
                arrowprops=dict(arrowstyle='->', color='#AAA', lw=0.8))

    ax.text(2.0, 1.0, 'Dynamic\n(per batch,\nrecent-weighted)', fontsize=8, ha='center', color=COLORS['accent'],
            bbox=dict(facecolor='#F0FFF4', edgecolor='#C3E6CB', boxstyle='round,pad=0.25'))
    ax.annotate('', xy=(3.0, 1.45), xytext=(2.2, 1.0),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=0.8))

    ax.text(7.15, 4.4, r'$\alpha \in [0,1]$' + '\nlearnable scalar\n(init from 0.6)',
            fontsize=8, ha='center', color=COLORS['secondary'],
            bbox=dict(facecolor='#FFF9E6', edgecolor='#FDEBD0', boxstyle='round,pad=0.25'))
    ax.annotate('', xy=(6.8, 3.4), xytext=(7.1, 4.15),
                arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=0.8))

    save(fig, 'fig_wind_adjacency')


# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 4 — RESULTS FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def fig_ablation_waterfall():
    """Progressive MAE improvement from baseline to confirmed best."""
    models = [
        'Historical\nMean',
        'Persistence',
        'MLP',
        'LSTM',
        'GCN-LSTM\n(static adj)',
        'GCN-LSTM\n(dynamic adj)',
        'GT + GCN',
        'GT + GATv1',
        'GT + GATv1\n+ Residual',
        'GT + log1p\n+ StdScaler\n+ Bias',
        'Temporal-First\n+ SegMoE\n(confirmed best)',
    ]
    mae_vals = [44.41, 23.98, 26.31, 22.52, 22.50, 21.64, 21.69, 21.18, 20.62, 19.79, 19.38]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(models))
    colors = [COLORS['neutral']] * (len(models) - 1) + [COLORS['primary']]
    colors[0] = COLORS['red']   # historical mean worst
    colors[1] = '#E07B39'       # persistence

    bars = ax.bar(x, mae_vals, color=colors, width=0.65, edgecolor='white', linewidth=0.8)

    # Value labels
    for bar, val in zip(bars, mae_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8.0, fontweight='bold')

    # Delta annotations for key steps
    key_steps = [(8, 9, '−0.83'), (9, 10, '−0.41')]
    for i1, i2, delta in key_steps:
        y_top = max(mae_vals[i1], mae_vals[i2]) + 1.5
        ax.annotate('', xy=(x[i2], mae_vals[i2] + 0.3), xytext=(x[i1], mae_vals[i1] + 0.3),
                    arrowprops=dict(arrowstyle='<->', color='#555', lw=1.0))
        ax.text((x[i1] + x[i2])/2, max(mae_vals[i1], mae_vals[i2]) + 1.0,
                delta, ha='center', fontsize=8.5, color='#333')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8.0)
    ax.set_ylabel('Test MAE (µg/m³)')
    ax.set_title('Progressive MAE Improvement — Architecture Ablation')
    ax.set_ylim(0, 52)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    # Legend
    legend_patches = [
        mpatches.Patch(color=COLORS['red'],     label='Non-parametric baselines'),
        mpatches.Patch(color=COLORS['neutral'], label='Intermediate models'),
        mpatches.Patch(color=COLORS['primary'], label='Confirmed best (multi-seed)'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=9)
    plt.tight_layout()
    save(fig, 'fig_ablation_waterfall')


def fig_model_comparison():
    """Final model comparison: top 8 deployable models + oracle."""
    models_short = [
        'GCN-LSTM\nFinal',
        'GT+GCN',
        'GT+GATv1',
        '+Residual',
        '+log1p\n+StdScaler',
        '+Bias\n(spatial-first)',
        'Temporal-\nFirst',
        'SegMoE\n(best ✓)',
        'Oracle\nFuture Met',
    ]
    mae  = [21.64, 21.69, 21.18, 20.62, 19.81, 19.79, 19.49, 19.38, 17.27]
    rmse = [39.05, 39.08, 38.07, 37.73, 37.51, 37.48, 37.27, 36.91, 32.83]

    x = np.arange(len(models_short))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.5))
    c_mae  = [COLORS['primary'] if i < len(models_short)-1 else COLORS['neutral']
              for i in range(len(models_short))]
    c_mae[-2] = '#1A6B3A'   # SegMoE best
    c_rmse = [COLORS['secondary'] if i < len(models_short)-1 else COLORS['neutral']
               for i in range(len(models_short))]
    c_rmse[-2] = COLORS['accent']

    bars_mae  = ax.bar(x - width/2, mae,  width, label='Test MAE',  color=c_mae,  edgecolor='white', lw=0.8)
    bars_rmse = ax.bar(x + width/2, rmse, width, label='Test RMSE', color=c_rmse, edgecolor='white', lw=0.8)

    for bar, val in zip(bars_mae, mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7.5, rotation=90)
    for bar, val in zip(bars_rmse, rmse):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(models_short, fontsize=8.5)
    ax.set_ylabel('Error (µg/m³)')
    ax.set_title('Model Comparison — Test MAE and RMSE')
    ax.set_ylim(0, 47)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=9)

    # Oracle annotation
    ax.axvline(x[-1] - width, color='#AAA', lw=0.8, linestyle=':')
    ax.text(x[-1], 44.5, 'Oracle\n(not deployable)', ha='center', fontsize=8, color='#888')

    plt.tight_layout()
    save(fig, 'fig_model_comparison')


def fig_per_horizon_mae():
    """Per-horizon MAE for 5 key models (grouped bar chart)."""
    horizons = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
    h = np.arange(len(horizons))

    data = {
        'GCN-LSTM Final':          [10.90, 16.05, 20.07, 23.57, 26.53, 29.08],
        'GT+GATv1+Residual':       [10.45, 15.99, 19.81, 22.99, 25.89, 28.60],
        'Spatial-First (Bias)':    [ 9.94, 15.07, 18.80, 22.12, 25.06, 27.77],
        'Temporal-First':          [ 9.77, 14.81, 18.59, 21.84, 24.74, 27.51],
        'SegMoE (best ✓)':         [ 9.61, 14.73, 18.43, 21.63, 24.58, 27.29],
        'Oracle Future Met':       [ 9.36, 14.03, 17.05, 19.34, 21.14, 22.71],
    }
    model_colors = [COLORS['neutral'], COLORS['teal'], COLORS['secondary'],
                    COLORS['primary'], '#1A6B3A', '#888888']
    model_styles = ['-', '-', '-', '-', '-', '--']

    fig, ax = plt.subplots(figsize=(9, 5))
    n = len(data)
    bw = 0.12
    offsets = np.linspace(-(n-1)*bw/2, (n-1)*bw/2, n)

    for i, (model, mae_h, color) in enumerate(zip(data.keys(), data.values(), model_colors)):
        ax.bar(h + offsets[i], mae_h, width=bw*0.92, label=model,
               color=color, alpha=0.85, edgecolor='white', lw=0.5)

    ax.set_xticks(h)
    ax.set_xticklabels(horizons)
    ax.set_ylabel('MAE (µg/m³)')
    ax.set_xlabel('Forecast Horizon')
    ax.set_title('Per-Horizon MAE Comparison — Key Models')
    ax.legend(fontsize=8.5, ncol=2, loc='upper left')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save(fig, 'fig_per_horizon_mae')


SEGMOE_PRED = os.path.join(
    os.path.dirname(__file__),
    'models', 'checkpoints', 'transformer', 'eval_predictions',
    'graph_transformer_gat_v1_residual_log1p_all_std_stationbias_temporal_first_SEgmoe_T4_best_predictions.npy',
)
SEGMOE_TARG = os.path.join(
    os.path.dirname(__file__),
    'models', 'checkpoints', 'transformer', 'eval_predictions',
    'graph_transformer_gat_v1_residual_log1p_all_std_stationbias_temporal_first_SEgmoe_T4_best_targets.npy',
)


def fig_prediction_timeseries():
    """Predicted vs ground truth PM2.5 at one station over 7-day test window."""
    preds   = np.load(SEGMOE_PRED)
    targets = np.load(SEGMOE_TARG)

    STATION_NAMES = [
        'Aotizhongxin', 'Changping', 'Dingling', 'Dongsi',
        'Guanyuan', 'Gucheng', 'Huairou', 'Nongzhanguan',
        'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong',
    ]
    # Show Dongsi (idx=3) and Dingling (idx=2)
    station_idxs = [3, 2]
    station_names_show = ['Dongsi (Urban)', 'Dingling (Suburban)']

    # Use H1 predictions (most direct comparison) for a 10-day window
    window = slice(0, 240)   # 240 hours = 10 days

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, sidx, sname in zip(axes, station_idxs, station_names_show):
        gt   = targets[window, 0, sidx]   # H1 ground truth
        pred = preds[window, 0, sidx]     # H1 prediction
        t    = np.arange(len(gt))

        ax.plot(t, gt,   color='#444', lw=1.4, label='Ground Truth', zorder=3)
        ax.plot(t, pred, color=COLORS['primary'], lw=1.4, linestyle='--',
                label='H1 Prediction', alpha=0.9, zorder=3)
        ax.fill_between(t, gt, pred, alpha=0.12, color=COLORS['primary'])

        mae_shown = float(np.mean(np.abs(gt - pred)))
        ax.set_ylabel('PM2.5 (µg/m³)')
        ax.set_title(f'{sname}  |  H1 MAE = {mae_shown:.2f} µg/m³')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_ylim(bottom=0)
        ax.yaxis.grid(True, linestyle='--', alpha=0.35)

    axes[-1].set_xlabel('Test Set Hour Index')
    plt.suptitle('Predicted vs. Ground Truth PM2.5 — 10-Day Test Window', fontsize=12, y=1.01)
    plt.tight_layout()
    save(fig, 'fig_prediction_timeseries')


def fig_per_station_mae():
    """Per-station MAE bar chart computed from saved predictions."""
    preds   = np.load(SEGMOE_PRED)
    targets = np.load(SEGMOE_TARG)

    STATION_NAMES = [
        'Aotizhongxin', 'Changping', 'Dingling', 'Dongsi',
        'Guanyuan', 'Gucheng', 'Huairou', 'Nongzhanguan',
        'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong',
    ]

    # MAE averaged over all horizons and test samples
    per_station_mae = np.mean(np.abs(preds - targets), axis=(0, 1))  # (12,)

    order = np.argsort(per_station_mae)
    sorted_names = [STATION_NAMES[i] for i in order]
    sorted_mae   = per_station_mae[order]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [COLORS['accent'] if m < np.median(sorted_mae) else COLORS['secondary']
              for m in sorted_mae]
    bars = ax.barh(range(len(sorted_names)), sorted_mae, color=colors,
                   edgecolor='white', lw=0.8)

    for bar, val in zip(bars, sorted_mae):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=8.5)

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel('Mean Absolute Error (µg/m³)')
    ax.set_title('Per-Station MAE — All Horizons Averaged')
    ax.axvline(np.mean(sorted_mae), color=COLORS['primary'], lw=1.2, linestyle='--',
               label=f'Network mean: {np.mean(sorted_mae):.2f}')
    ax.legend(fontsize=9)
    ax.xaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save(fig, 'fig_per_station_mae')


def fig_baseline_horizon_crossover():
    """
    Per-horizon MAE: Persistence vs MLP.
    Shows that MLP is worse at H1-H4 (strong autocorrelation) but catches up
    at H5-H6 once lag-1 copying degrades — the crossover pattern.
    Built entirely from hardcoded verified numbers (re-evaluated 2026-05-07).
    """
    horizons = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
    persistence_mae = [10.619, 17.376, 22.685, 27.245, 31.224, 34.748]
    mlp_mae         = [19.149, 21.651, 24.761, 27.909, 30.901, 33.506]

    x = np.arange(len(horizons))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_p = ax.bar(x - width/2, persistence_mae, width, label='Persistence',
                    color=COLORS['secondary'], edgecolor='white', lw=0.8)
    bars_m = ax.bar(x + width/2, mlp_mae,         width, label='MLP (flattened)',
                    color=COLORS['primary'],   edgecolor='white', lw=0.8, alpha=0.85)

    for bar, val in zip(bars_p, persistence_mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars_m, mlp_mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    # Annotate H1 delta
    ax.annotate('', xy=(x[0] + width/2, mlp_mae[0] + 0.5),
                xytext=(x[0] - width/2, persistence_mae[0] + 0.5),
                arrowprops=dict(arrowstyle='<->', color=COLORS['red'], lw=1.3))
    ax.text(x[0], max(persistence_mae[0], mlp_mae[0]) + 2.5,
            f'+{mlp_mae[0]-persistence_mae[0]:.1f}', ha='center', fontsize=8.5,
            color=COLORS['red'], fontweight='bold')

    # Crossover annotation band
    ax.axvspan(3.5, 5.5, alpha=0.07, color=COLORS['accent'])
    ax.text(4.5, 2.5, 'MLP wins\n(H5–H6)', ha='center', fontsize=8.5,
            color=COLORS['accent'], style='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_ylabel('MAE (µg/m³)')
    ax.set_title('Persistence vs. MLP — Per-Horizon MAE\n'
                 '(Strong PM2.5 autocorrelation favours persistence at H1–H4)')
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 42)
    plt.tight_layout()
    save(fig, 'fig_baseline_horizon_crossover')


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f'Generating thesis figures -> {OUT}\n')

    print('Chapter 2 — Background:')
    fig_lstm_cell()
    fig_transformer_encoder()
    fig_gat_attention()

    print('\nChapter 3 — Methodology:')
    fig_station_map()
    fig_pm25_timeseries()
    fig_model_architecture()
    fig_segmoe_block()
    fig_wind_adjacency()

    print('\nChapter 4 — Results:')
    fig_ablation_waterfall()
    fig_model_comparison()
    fig_per_horizon_mae()
    fig_prediction_timeseries()
    fig_per_station_mae()
    fig_baseline_horizon_crossover()

    print(f'\nDone. {len(os.listdir(OUT))//2} figures saved to {OUT}')
