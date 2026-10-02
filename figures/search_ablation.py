"""Search ablation. Run with python -m figures.search_ablation."""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from .style import apply_medagents_style, get_figure_3_colors, save_figure


def plot_search_modality_comparison(ax, df, colors, panel_label='a'):
    """Compare different search modalities (web vs vector vs both)"""

    # Configuration
    order_keys = ['both', 'vector_only', 'web_only', 'none', 'random']
    modality_names = ['Web & Corpus', 'Corpus',
                      'Web', 'Disabled', 'Random']

    # Filter data for search modality ablation
    modality_df = df[df['ablation'] == 'search_modality']

    if modality_df.empty:
        raise ValueError("No 'search_modality' experimental data found in provided DataFrame for Figure 3A.")

    # Assemble arrays in a fixed order
    accuracies, times = [], []
    for key in order_keys:
        row = modality_df[modality_df['exp_name'] == key]
        if not row.empty:
            accuracies.append(float(row['accuracy'].values[0]))
            times.append(float(row['avg_time'].values[0]))
        else:
            accuracies.append(np.nan)
            times.append(np.nan)

    x = np.arange(len(order_keys))
    color_list = [colors['modality'].get(k, colors['metrics']['accuracy']) for k in order_keys]
    bars = ax.bar(x, accuracies, 0.6, color=color_list, alpha=0.9, edgecolor='black', linewidth=1.5, label='Accuracy (%)')

    # Accuracy labels
    for xi, bar, acc in zip(x, bars, accuracies):
        if np.isfinite(acc):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.6,
                    f'{acc:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Time line on secondary axis
    ax_time = ax.twinx()
    valid = np.isfinite(times)
    ax_time.plot(x[valid], np.array(times)[valid], color=colors['metrics']['time'], marker='o', linewidth=2.0, label='Time (s)')

    # Styling
    ax.set_xlabel('Search Modality', fontweight='bold', fontsize=13, color='black')
    ax.set_ylabel('Accuracy (%)', fontweight='bold', color='black', fontsize=11)
    ax.set_xticks(np.arange(len(modality_names)))
    ax.set_xticklabels(modality_names, fontsize=10)
    ax.tick_params(axis='both', labelsize=10, colors='black')
    ax_time.set_ylabel('Time (s)', fontweight='bold', color='black', fontsize=11)
    ax_time.tick_params(axis='y', labelsize=10, colors='black')
    # Dynamic y-limits
    try:
        max_acc = np.nanmax(accuracies) if 'accuracies' in locals() else 40
        ax.set_ylim(15, max(40, max_acc + 6))
    except Exception:
        ax.set_ylim(15, 40)
    try:
        max_time = np.nanmax(times) if 'times' in locals() else 400
        ax_time.set_ylim(0, max_time * 1.2)
    except Exception:
        ax_time.set_ylim(0, 400)

    # Panel label
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=22, fontweight='bold')

    # Legend combining bar and line
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_time.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='upper center',
        bbox_to_anchor=(0.5, 1),
        ncol=2,
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        facecolor='white',
        edgecolor='black',
    )
    return ax, ax_time


def plot_search_features_comparison(ax, df, colors, panel_label='b'):
    """Compare search features ablation"""

    # Filter data for search features ablation
    features_df = df[df['ablation'] == 'search_features']

    feature_mapping = {
        'baseline': 'Full',
        'no_document_review': '+ Query Rewrite',
        'no_query_rewrite': '+ Doc Review',
        'no_rewrite_no_review': 'Search Only'
    }

    if features_df.empty:
        raise ValueError("No 'search_features' experimental data found in provided DataFrame for Figure 3B.")

    # Assemble arrays in fixed order
    order_keys = ['baseline', 'no_document_review', 'no_query_rewrite', 'no_rewrite_no_review']
    names = [feature_mapping[k] for k in order_keys]
    acc, times = [], []
    for k in order_keys:
        row = features_df[features_df['exp_name'] == k]
        if not row.empty:
            acc.append(float(row['accuracy'].values[0]))
            times.append(float(row['avg_time'].values[0]))
        else:
            acc.append(np.nan)
            times.append(np.nan)
    x = np.arange(len(order_keys))
    color_list = [colors['features'].get(k, colors['metrics']['accuracy']) for k in order_keys]
    bars = ax.bar(x, acc, color=color_list, alpha=0.85, edgecolor='black', linewidth=2, width=0.6, label='Accuracy (%)')

    # Accuracy labels above bars
    for xi, bar, a in zip(x, bars, acc):
        if np.isfinite(a):
            ax.annotate(f'{a:.0f}%', (bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Time line
    ax_time = ax.twinx()
    valid = np.isfinite(times)
    ax_time.plot(x[valid], np.array(times)[valid], color=colors['metrics']['time'], marker='o', linewidth=2.0, label='Time (s)')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_xlabel('Search Features', fontweight='bold', fontsize=13)

    # Styling
    # Hide left y-axis (accuracy) for panel B
    ax.set_ylabel('')
    ax.tick_params(axis='both', labelsize=10)
    ax.tick_params(left=False, labelleft=False)
    try:
        ax.spines['left'].set_visible(False)
    except Exception:
        pass
    ax.grid(True, alpha=0.3, axis='y', linewidth=0.5)
    # Dynamic y-limits
    try:
        max_acc = np.nanmax(acc) if 'acc' in locals() else max(accuracies)
        ax.set_ylim(15, max(40, max_acc + 6))
    except Exception:
        ax.set_ylim(15, 40)
    try:
        max_time = np.nanmax(times)
        ax_time.set_ylabel('Time (s)', fontweight='bold', fontsize=11)
        ax_time.tick_params(axis='y', labelsize=10)
        ax_time.set_ylim(0, max_time * 1.2)
    except Exception:
        pass

    # Panel label
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=22, fontweight='bold')

    # Legend combining bar and line
    try:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_time.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc='upper center',
            bbox_to_anchor=(0.5, 1),
            ncol=2,
            fontsize=12,
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.95,
            facecolor='white',
            edgecolor='black',
        )
    except Exception:
        pass
    return ax


def plot_search_history_comparison(ax, df, colors, panel_label='c'):
    """Compare search history strategies"""

    # Filter data for search history ablation
    history_df = df[df['ablation'] == 'search_history']

    if history_df.empty:
        raise ValueError("No 'search_history' experimental data found in provided DataFrame for Figure 3C.")

    history_mapping = {
        'individual': 'Individual Search',
        'shared': 'Shared Search'
    }
    order_keys = ['individual', 'shared']
    names = [history_mapping[k] for k in order_keys]
    acc, times = [], []
    for k in order_keys:
        row = history_df[history_df['exp_name'] == k]
        if not row.empty:
            acc.append(float(row['accuracy'].values[0]))
            times.append(float(row['avg_time'].values[0]))
        else:
            acc.append(np.nan)
            times.append(np.nan)
    x = np.arange(len(order_keys))
    colors_list = [colors['history'].get(k, colors['metrics']['accuracy']) for k in order_keys]
    bars = ax.bar(x, acc, color=colors_list, alpha=0.85, edgecolor='black', linewidth=2, width=0.6, label='Accuracy (%)')

    # Accuracy labels above bars
    for xi, bar, a in zip(x, bars, acc):
        if np.isfinite(a):
            ax.annotate(f'{a:.0f}%', (bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5),
                        ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Time line
    ax_time = ax.twinx()
    valid = np.isfinite(times)
    ax_time.plot(x[valid], np.array(times)[valid], color=colors['metrics']['time'], marker='o', linewidth=2.0, label='Time (s)')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_xlabel('Search History Strategy', fontweight='bold', fontsize=13)

    # Styling
    # Hide left y-axis (accuracy) for panel C
    ax.set_ylabel('')
    ax.tick_params(axis='both', labelsize=10)
    ax.tick_params(left=False, labelleft=False)
    try:
        ax.spines['left'].set_visible(False)
    except Exception:
        pass
    ax.grid(True, alpha=0.3, axis='y', linewidth=0.5)
    try:
        max_acc = np.nanmax(acc) if 'acc' in locals() else max(accuracies)
        ax.set_ylim(15, max(40, max_acc + 6))
    except Exception:
        ax.set_ylim(15, 40)
    try:
        max_time = np.nanmax(times)
        ax_time.set_ylabel('Time (s)', fontweight='bold', fontsize=11)
        ax_time.tick_params(axis='y', labelsize=10)
        ax_time.set_ylim(0, max_time * 1.2)
    except Exception:
        pass

    # Panel label
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=22, fontweight='bold')

    # Legend combining bar and line
    try:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_time.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc='upper center',
            bbox_to_anchor=(0.5, 1),
            ncol=2,
            fontsize=12,
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.95,
            facecolor='white',
            edgecolor='black',
        )
    except Exception:
        pass
    return ax


def create_figure_3():
    """Create Figure 3: Search Ablation Analysis"""

    apply_medagents_style()

    colors = get_figure_3_colors()

    data_path = os.path.join(os.path.dirname(__file__), 'data', 'search_ablation.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Expected experimental data at '{data_path}' for Figure 3."
        )
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} search ablation records from experimental data")

    fig = plt.figure(figsize=(18, 5))
    # Make A wider and C narrower via width ratios
    gs = fig.add_gridspec(1, 3, height_ratios=[1], width_ratios=[1.6, 1.2, 1.0],
                          hspace=0.1, wspace=0.15)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_search_modality_comparison(ax1, df, colors, panel_label='a')

    ax2 = fig.add_subplot(gs[0, 1])
    plot_search_features_comparison(ax2, df, colors, panel_label='b')

    ax3 = fig.add_subplot(gs[0, 2])
    plot_search_history_comparison(ax3, df, colors, panel_label='c')

    plt.tight_layout()
    save_figure(fig, 'fig-s1.search_ablation.pdf')

    return fig


if __name__ == "__main__":
    create_figure_3()
