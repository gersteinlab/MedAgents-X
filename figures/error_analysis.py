"""Error analysis. Run with python -m figures.error_analysis."""
import json
import matplotlib.pyplot as plt
import numpy as np
from .style import apply_medagents_style, get_manchester_colors, save_figure
from collections import Counter
from pathlib import Path


REASONING_DATASETS = {"medqa", "medbullets", "medxpertqa-r"}


CATEGORY_ORDER = [
    "Misled Discussion",
    "Evidence Error",
    "Reasoning Error",
    "Insufficient Discussion",
]


mc = get_manchester_colors()


COLORS = {
    "Misled Discussion": mc['penn_red'],
    "Evidence Error": mc['sandy_brown_light'],
    "Reasoning Error": mc['persimmon'],
    "Insufficient Discussion": mc['gray'],
}


LABEL_MAP = {
    "Ignored Answer": "Misled Discussion",
    "Overconfidence": "Misled Discussion",
    "Faulty Reasoning": "Reasoning Error",
    "Poor Evidence": "Evidence Error",
    "Premature Termination": "Insufficient Discussion",
}


def _load_counts(err_dir: Path) -> dict:
    # Primary expected path
    path = err_dir / "counts_by_run.json"
    if path.exists():
        return json.loads(path.read_text())
    # Fallback for alternate naming
    alt = err_dir / "count_by_run.json"
    if alt.exists():
        return json.loads(alt.read_text())
    raise FileNotFoundError(f'No error counts in {err_dir}')


def _aggregate(counts_by_run: dict):
    reasoning = Counter()
    knowledge = Counter()
    runs_r = 0
    runs_k = 0
    for run_key, cat_counts in counts_by_run.items():
        dataset = run_key.split("/")[0] if "/" in run_key else run_key
        if dataset in REASONING_DATASETS:
            target = reasoning
            runs_r += 1
        else:
            target = knowledge
            runs_k += 1
        for cat, cnt in cat_counts.items():
            if cat == "Success":
                continue
            # Map to grouped categories; fall back to original if unseen
            new_cat = LABEL_MAP.get(cat, cat)
            try:
                target[new_cat] += int(cnt)
            except Exception:
                pass
    return reasoning, knowledge, runs_r, runs_k


def plot_error_pies(ax_left, ax_right, base_dir: Path, panel_label_left='a', panel_label_right='b'):
    data = _load_counts(base_dir)
    if not data:
        for ax in (ax_left, ax_right):
            ax.axis('off')
            ax.text(0.5, 0.5, 'No error counts', ha='center', va='center')
        return
    r_counts, k_counts, runs_r, runs_k = _aggregate(data)
    labels = [c for c in CATEGORY_ORDER if (r_counts.get(c, 0) + k_counts.get(c, 0)) > 0]

    def draw_donut(ax, sizes, labels, title, panel_label=None):
        total = sum(sizes)
        if total == 0:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No errors', ha='center', va='center')
            return
        colors = [COLORS.get(l, "#999999") for l in labels]
        # Donut chart: no on-slice labels to avoid clutter
        wedges, _ = ax.pie(
            sizes,
            startangle=90,
            colors=colors,
            labels=None,
            wedgeprops=dict(width=0.3, edgecolor='white'),
            radius=1.2
        )
        # Center total count
        ax.text(0, 0, f"N={total}", ha='center', va='center', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=18, fontweight='bold')
        if panel_label:
            ax.text(-0.3, 1.02, panel_label, transform=ax.transAxes, ha='left', va='bottom', fontsize=22, fontweight='bold')
        # Legend with counts and percentages
        legend_items = []
        for l, s in zip(labels, sizes):
            pct = (s / total * 100.0) if total > 0 else 0.0
            legend_items.append(f"{l} ({pct:.0f}%)")
        # Put legend below the chart; 2 columns if many categories
        ncol = 2 if len(legend_items) > 3 else 1
        ax.legend(
            wedges,
            legend_items,
            loc='upper center',
            bbox_to_anchor=(0.5, 0),
            frameon=False,
            ncol=ncol,
            fontsize=10,
            handlelength=1.2,
            handletextpad=0.6,
            columnspacing=0.9,
        )

    draw_donut(ax_left, [r_counts.get(label, 0) for label in labels], labels,
               'Errors — Reasoning', panel_label_left)
    draw_donut(ax_right, [k_counts.get(label, 0) for label in labels], labels,
               'Errors — Knowledge', panel_label_right)


def plot_error_agreement_heatmap(ax, base_dir: Path, panel_label='c'):
    path = base_dir / "annotator_agreement.json"
    data = json.loads(path.read_text())
    a1 = data.get("annotations1", [])
    a2 = data.get("annotations2", [])
    if not a1 or not a2 or len(a1) != len(a2):
        ax.axis('off'); ax.text(0.5, 0.5, 'Missing annotations', ha='center', va='center'); return
    # Remap to grouped categories
    a1 = [LABEL_MAP.get(x, x) for x in a1]
    a2 = [LABEL_MAP.get(x, x) for x in a2]
    # Ensure core categories always appear even if zero-count
    base_cats = ['Misled Discussion', 'Evidence Error', 'Reasoning Error', 'Insufficient Discussion']
    cats_set = set(a1 + a2)
    cats = [c for c in base_cats] + sorted(list(cats_set - set(base_cats)))
    # Abbreviations for category labels
    abbr = {
        'Misled Discussion': 'MD',
        'Evidence Error': 'EE',
        'Reasoning Error': 'RE',
        'Insufficient Discussion': 'ID',
        'UNKNOWN': 'UNK',
    }
    cats_short = [abbr.get(c, c) for c in cats]
    idx = {c: i for i, c in enumerate(cats)}
    M = np.zeros((len(cats), len(cats)), dtype=int)
    for x, y in zip(a1, a2):
        M[idx[x], idx[y]] += 1
    # Use MedAgents colorset for a sequential colormap (white -> intense)
    mc = get_manchester_colors()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        'medagents_seq', ['white', mc['sandy_brown_light'], mc['penn_red']]
    )
    im = ax.imshow(M, cmap=cmap)
    ax.set_aspect('equal')
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats_short, rotation=45, ha='right')
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats_short)
    ax.set_xlabel("Annotator B", fontweight='bold', fontsize=14)
    ax.set_ylabel("Annotator A", fontweight='bold', fontsize=14)
    ax.tick_params(axis='both', labelsize=14, length=0)
    kappa = data.get("cohen_kappa", 0.0)
    ax.set_title(f"Inter-Annotator Agreement (κ={kappa:.2f})", fontsize=14, fontweight='bold')
    ax.text(-0.3, 1.02, panel_label, transform=ax.transAxes, ha='left', va='bottom', fontsize=22, fontweight='bold')
    for i in range(len(cats)):
        for j in range(len(cats)):
            v = M[i, j]
            if v:
                # Determine text color based on background darkness
                bg_color = cmap(M[i, j] / M.max() if M.max() > 0 else 0)
                # Calculate luminance to determine if background is dark
                luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
                text_color = 'white' if luminance < 0.5 else mc['black']
                ax.text(j, i, str(v), va='center', ha='center', fontsize=12, color=text_color)
    # Colorbar to the right
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    import matplotlib.pyplot as plt
    cb = plt.colorbar(im, cax=cax)
    cb.set_label('Count', fontsize=14, fontweight='bold')
    cb.ax.tick_params(labelsize=12)


def plot_evidence_pyramid_grid(axes_grid, base_dir: Path, panel_label='d'):
    """3x3 grid: rows are retrieval modalities (both, vector_only, web_only),
    columns are criteria (reliability, relevance, helpfulness). Each cell is a
    pyramid comparing Annotator A vs B score distributions (1–5).
    axes_grid should be a 3x3 array-like of Axes.
    """
    path = base_dir / "evidence_quality_scores.json"
    data = json.loads(path.read_text())  # modality -> list of entries
    modalities = ['both', 'vector_only', 'web_only']
    crits = ['reliability', 'relevance', 'helpfulness']
    # counts[modality][criterion][annotator][score]
    from collections import defaultdict
    counts = {m: {c: {"A": defaultdict(int), "B": defaultdict(int)} for c in crits} for m in modalities}
    # means[criterion][modality] -> average score across both annotators (if available)
    means = {c: {m: None for m in modalities} for c in crits}
    sum_scores = {c: {m: 0.0 for m in modalities} for c in crits}
    n_scores = {c: {m: 0 for m in modalities} for c in crits}
    for m in modalities:
        for e in data.get(m, []) or []:
            sA = (e.get("scores") or {})
            sB = (e.get("scores_alt") or {})
            for c in crits:
                vA = sA.get(c); vB = sB.get(c)
                if isinstance(vA, int) and 1 <= vA <= 5: counts[m][c]["A"][vA] += 1
                if isinstance(vB, int) and 1 <= vB <= 5: counts[m][c]["B"][vB] += 1
                # accumulate for mean: average across available annotators
                for v in (vA, vB):
                    if isinstance(v, int) and 1 <= v <= 5:
                        sum_scores[c][m] += v
                        n_scores[c][m] += 1
    for c in crits:
        for m in modalities:
            if n_scores[c][m] > 0:
                means[c][m] = sum_scores[c][m] / float(n_scores[c][m])

    labels = ["1", "2", "3", "4", "5"]
    y = np.arange(len(labels))
    mc = get_manchester_colors()
    colors = {"A": mc['penn_red'], "B": mc['jet']}
    modality_titles = {"both": "Both Retrieval", "vector_only": "Corpus-only", "web_only": "Web-only"}

    # Titles for top row columns; append per-modality means for Reliability, Relevance, Helpfulness
    order = ["both", "vector_only", "web_only"]
    label_map = {"both": "B", "vector_only": "C", "web_only": "W"}
    for j, crit in enumerate(crits):
        ax_top = axes_grid[0][j]
        title = crit.capitalize()
        if crit in ("reliability", "relevance", "helpfulness"):
            vals = []
            for m in order:
                mu = means[crit][m]
                vals.append(f"{mu:.1f}" if mu is not None else "-")
            title = f"{title} ({'/'.join(vals)})"
        ax_top.set_title(title, fontsize=16, fontweight='bold')

    for i, m in enumerate(modalities):
        for j, c in enumerate(crits):
            ax = axes_grid[i][j]
            left_raw = [counts[m][c]["A"].get(k, 0) for k in range(1, 6)]
            right_raw = [counts[m][c]["B"].get(k, 0) for k in range(1, 6)]
            left = [-v for v in left_raw]; right = right_raw
            bars_left = ax.barh(y, left, color=colors["A"], alpha=0.85, label="Annotator A")
            bars_right = ax.barh(y, right, color=colors["B"], alpha=0.85, label="Annotator B")
            ax.set_yticks(y)
            if j == 0:
                ax.set_yticklabels(labels, fontsize=12)
            else:
                ax.set_yticklabels([])
            mmax = max(max(abs(v) for v in left) if left else 1, max(right) if right else 1, 1)
            ax.set_xlim(-mmax * 1.15, mmax * 1.15)
            ax.axvline(0, color="#666", linewidth=1)
            ax.set_xticks([-mmax, 0, mmax])
            if i == len(modalities) - 1:
                ax.set_xticklabels(["Annotator A", "", "Annotator B"], fontsize=12)
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis='x', length=0)
            # Row titles on the left
            if j == 0:
                title = modality_titles[m]
                if i == 1:
                    ax.set_ylabel(f"Score (1-5)\n{title}", fontweight='bold', fontsize=12)
                else:
                    ax.set_ylabel(title, fontweight='bold', fontsize=12)
            # Annotate counts lightly
            for rect in bars_right:
                w = rect.get_width();
                if w and mmax > 0:
                    ax.text(w + mmax * 0.015, rect.get_y() + rect.get_height()/2, str(int(w)), va='center', ha='left', fontsize=8)
            for rect in bars_left:
                w = abs(rect.get_width());
                if w and mmax > 0:
                    ax.text(-w - mmax * 0.015, rect.get_y() + rect.get_height()/2, str(int(w)), va='center', ha='right', fontsize=8)
    # Panel label on the top-left subplot
    axes_grid[0][0].text(-0.08, 1.15, panel_label, transform=axes_grid[0][0].transAxes, ha='left', va='bottom', fontsize=22, fontweight='bold')


def create_figure_2():
    apply_medagents_style()

    root = Path(__file__).resolve().parent / 'data'
    err_dir = root / 'error_analysis'
    eq_dir = root / 'evidence_quality'

    fig = plt.figure(figsize=(18, 10))
    # 2 rows: top has 3 columns (A,B,C), bottom row (D) spans all columns with a 1x3 subgrid
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.2], width_ratios=[1, 1, 1], hspace=0.32, wspace=0.22)

    # Top row: A (Reasoning pie), B (Knowledge pie), C (Heatmap)
    ax_pie_r = fig.add_subplot(gs[0, 0])
    ax_pie_k = fig.add_subplot(gs[0, 1])
    plot_error_pies(ax_pie_r, ax_pie_k, err_dir, panel_label_left='a', panel_label_right='b')

    ax_heat = fig.add_subplot(gs[0, 2])
    plot_error_agreement_heatmap(ax_heat, err_dir, panel_label='c')

    # Bottom row: d (Pyramid) as 3x3 subgrid spanning all columns
    subgs = gs[1, :].subgridspec(3, 3, hspace=0.25, wspace=0.25)
    axes_grid = [[fig.add_subplot(subgs[i, j]) for j in range(3)] for i in range(3)]
    plot_evidence_pyramid_grid(axes_grid, eq_dir, panel_label='d')

    plt.tight_layout()
    out_pdf = 'fig-4.error_analysis_overview.pdf'
    save_figure(fig, out_pdf)
    print(f"[figure-4] Wrote: {out_pdf}")
    return fig


if __name__ == "__main__":
    create_figure_2()
