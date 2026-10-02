"""Main comparison. Run with python -m figures.main_comparison."""
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from .style import apply_medagents_style, get_figure_0_colors, save_figure


def plot_overall_performance_bar(ax, df, colors, panel_label):
    """Create overall performance bar chart with time series overlay"""

    # Configuration
    MODELS = ['o3-mini', 'gpt-4o', 'gpt-4o-mini']
    METHODS = ['CoT', 'CoT-SC', 'MedPrompt', 'MultiPersona', 'MedAgents', 'AFlow',
               'MedAgents-X', 'Few-shot', 'MDAgents', 'SPO', 'Self-refine', 'MedRAG', 'Zero-shot']

    method_colors = colors.get('methods', {})

    # Calculate method statistics
    method_stats = {}
    for method in METHODS:
        method_df = df[df['method'] == method]
        if not method_df.empty:
            model_accuracies = {}
            model_times = {}
            for model in MODELS:
                model_data = method_df[method_df['model'] == model]
                if not model_data.empty:
                    model_accuracies[model] = {
                        'mean': model_data['accuracy'].mean(),
                        'std': model_data['accuracy'].std(),
                        'values': model_data['accuracy'].values
                    }
                    model_times[model] = {
                        'mean': model_data['avg_time'].mean(),
                        'std': model_data['avg_time'].std(),
                        'values': model_data['avg_time'].values
                    }
            method_stats[method] = {'accuracy': model_accuracies, 'time': model_times}

    # Sort methods by overall performance
    overall_means = []
    for method in METHODS:
        if method in method_stats:
            all_values = []
            for model in MODELS:
                if model in method_stats[method]['accuracy']:
                    all_values.extend(method_stats[method]['accuracy'][model]['values'])
            if all_values:
                overall_means.append((method, np.mean(all_values)))

    overall_means.sort(key=lambda x: x[1], reverse=True)
    methods_sorted = [item[0] for item in overall_means]

    # Plot bars
    bar_width = 0.3
    x_positions = np.arange(len(methods_sorted))
    alphas = [0.9, 0.7, 0.5]

    for i, model in enumerate(MODELS):
        model_means = []
        model_stds = []
        model_values_list = []

        for method in methods_sorted:
            if method in method_stats and model in method_stats[method]['accuracy']:
                model_means.append(method_stats[method]['accuracy'][model]['mean'])
                model_stds.append(method_stats[method]['accuracy'][model]['std'])
                model_values_list.append(method_stats[method]['accuracy'][model]['values'])
            else:
                model_means.append(0)
                model_stds.append(0)
                model_values_list.append([])

        x_pos = x_positions + (i - 1) * bar_width

        # Bars
        ax.bar(x_pos, model_means, bar_width,
               color=[method_colors.get(method, 'gray') for method in methods_sorted],
               alpha=alphas[i], edgecolor='black', linewidth=1.5, label=f'{model}')

        ax.errorbar(x_pos, model_means, yerr=model_stds, fmt='none',
                   ecolor='black', capsize=4, capthick=2, linewidth=2, alpha=0.8)

        for j, (x, mean, values) in enumerate(zip(x_pos, model_means, model_values_list)):
            if len(values) > 0:
                for value in values:
                    ax.plot(x, value, 'o', color='darkred' if i == 0 else 'darkblue',
                           markersize=4, alpha=0.7, markeredgecolor='black', markeredgewidth=0.5)

    ax_twin = ax.twinx()
    for i, model in enumerate(MODELS):
        model_time_means = []
        model_time_stds = []

        for method in methods_sorted:
            if method in method_stats and model in method_stats[method]['time']:
                model_time_means.append(method_stats[method]['time'][model]['mean'])
                model_time_stds.append(method_stats[method]['time'][model]['std'])
            else:
                model_time_means.append(0)
                model_time_stds.append(0)

        line_color = 'navy' if i == 0 else 'maroon'
        line_style = '-' if i == 0 else '--'
        marker_style = 'D' if i == 0 else '^'

        ax_twin.plot(x_positions, model_time_means, color=line_color, linestyle=line_style,
                    marker=marker_style, markersize=10, linewidth=4, alpha=0.9,
                    label=f'{model} (Time)', markeredgecolor='white', markeredgewidth=1)

        ax_twin.errorbar(x_positions, model_time_means, yerr=model_time_stds, fmt='none',
                        ecolor=line_color, capsize=3, capthick=2, linewidth=2, alpha=0.7)

    if methods_sorted:
        best_method_idx = 0
        best_method = methods_sorted[0]
        best_x = x_positions[best_method_idx]
        best_y = max([method_stats[best_method]['accuracy'][model]['mean']
                     for model in MODELS if model in method_stats[best_method]['accuracy']])

        ax.plot(best_x, best_y + 3, marker='*', markersize=25, color='gold',
               markeredgecolor='black', markeredgewidth=3, zorder=15, label='Best Overall')

    ax.set_ylim(8, 48)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=18)
    ax_twin.set_ylabel('Execution Time (seconds)', fontweight='bold', fontsize=18)
    ax.set_xlabel('Methods (Ranked by Performance)', fontweight='bold', fontsize=18)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(methods_sorted, fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    ax_twin.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.4, axis='y', linewidth=0.8, linestyle=':')

    # Legend
    model_legend_elements = []
    for i, model in enumerate(MODELS):
        model_legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=alphas[i],
                                                 edgecolor='black', linewidth=1.5, label=f'{model} (Accuracy)'))

    time_legend_elements = []
    for i, model in enumerate(MODELS):
        line_color = 'navy' if i == 0 else 'maroon'
        line_style = '-' if i == 0 else '--'
        marker_style = 'D' if i == 0 else '^'
        time_legend_elements.append(plt.Line2D([0], [0], color=line_color, linestyle=line_style,
                                             marker=marker_style, markersize=10, linewidth=4,
                                             label=f'{model} (Time)', markeredgecolor='white', markeredgewidth=1))

    special_legend_elements = [plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                                        markeredgecolor='black', markeredgewidth=3, markersize=25,
                                        label='Best Overall', linestyle='None')]

    combined_legend_elements = model_legend_elements + time_legend_elements + special_legend_elements
    ax.legend(handles=combined_legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98),
             title='Legend', fontsize=13, ncol=1, frameon=True, fancybox=True, shadow=True,
             framealpha=0.95, facecolor='white', edgecolor='black', title_fontsize=15)

    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=22, fontweight='bold')

    return ax, ax_twin


def plot_pareto_frontier(ax, df, colors, panel_label):
    """Create Pareto frontier cost vs accuracy plot"""

    method_colors = colors.get('methods', {})

    # Calculate average metrics
    avg_metrics = df.groupby('method').agg({
        'accuracy': 'mean',
        'avg_cost': lambda x: x.mean() * 100,  # Convert to cents
        'avg_time': 'mean'
    }).reset_index()

    # Plot scatter points (cost on x, accuracy on y)
    for _, row in avg_metrics.iterrows():
        if row['method'] == 'MedAgents-X':
            ax.scatter(row['avg_cost'], row['accuracy'],
                      c=method_colors.get(row['method'], 'gray'),
                      s=500, alpha=0.9,
                      edgecolors='black', linewidth=3,
                      label=row['method'], marker='*', zorder=10)
        else:
            ax.scatter(row['avg_cost'], row['accuracy'],
                      c=method_colors.get(row['method'], 'gray'),
                      s=250, alpha=0.8,
                      edgecolors='black', linewidth=2,
                      label=row['method'], zorder=5)

    # Calculate Pareto frontier
    pareto_indices = []
    sorted_indices = np.argsort(avg_metrics['avg_cost'])
    max_accuracy_so_far = -1
    for i in sorted_indices:
        if avg_metrics.iloc[i]['accuracy'] > max_accuracy_so_far:
            pareto_indices.append(i)
            max_accuracy_so_far = avg_metrics.iloc[i]['accuracy']

    if len(pareto_indices) > 1:
        pareto_data = avg_metrics.iloc[pareto_indices].sort_values('avg_cost')
        ax.plot(pareto_data['avg_cost'], pareto_data['accuracy'], 'k--', alpha=0.8, linewidth=3,
               label='Pareto Frontier', zorder=8)

    # Add annotations
    for _, row in avg_metrics.iterrows():
        offset_x = 12 if row['method'] != 'MedAgents-X' else 15
        offset_y = 10 if row['method'] != 'MedAgents-X' else 12
        ax.annotate(row['method'], (row['avg_cost'], row['accuracy']),
                   xytext=(offset_x, offset_y), textcoords='offset points',
                   fontsize=13, ha='left', va='bottom', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9,
                            edgecolor='gray', linewidth=1.5))

    # Styling
    ax.set_xlabel('Average Cost (cents per query)', fontweight='bold', fontsize=18)
    ax.set_ylabel('Average Accuracy (%)', fontweight='bold', fontsize=18)
    ax.set_ylim(14, 34)
    ax.tick_params(axis='both', labelsize=15)
    ax.grid(True, alpha=0.4, linestyle=':', linewidth=1)

    # Legend
    pareto_legend = [plt.Line2D([0], [0], color='black', linestyle='--', linewidth=3,
                               label='Pareto Frontier', alpha=0.8)]
    ax.legend(handles=pareto_legend, loc='lower right', fontsize=13,
             frameon=True, fancybox=True, shadow=True, framealpha=0.95,
             facecolor='white', edgecolor='black')

    # Panel label
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=22, fontweight='bold')

    return ax


def _parse_agent_cfg(exp_name: str):
    parts = exp_name.split('_')
    # Expect patterns like: "3_agents_1_round_with_search"
    n_agents = int(parts[0])
    n_rounds = int(parts[2])
    has_search = 'with_search' in exp_name
    return n_agents, n_rounds, has_search


def _mean_of(df, mask, col):
    s = df.loc[mask, col]
    return float(s.mean()) if len(s) else np.nan


def plot_component_breakdown_waterfall(ax, df, role_play_df, orchestration_df, colors, panel_label='c'):
    """Plot a simplified waterfall chart showing cumulative accuracy with time/cost overlays.

    - Bars: cumulative accuracy starting from baseline, then increments per component.
    - Lines (right axis): cumulative time and cost (cost scaled to per-100 for readability).
    """

    # Parse agent configuration columns once
    tmp = df['exp_name'].apply(_parse_agent_cfg).apply(pd.Series)
    tmp.columns = ['n_agents', 'n_rounds', 'has_search']
    df = pd.concat([df.copy(), tmp], axis=1)

    # Components in order
    components = [
        'Baseline',
        'Multi-Agent',
        'Evidence\nRetrieval',
        'Iterative\nReasoning',
        'Role Play',
        'Discussion\nOrchestration',
    ]

    # Baseline (1 agent, 1 round, no search)
    m_base_acc = _mean_of(df, (df.n_agents == 1) & (df.n_rounds == 1) & (~df.has_search), 'accuracy')
    m_base_time = _mean_of(df, (df.n_agents == 1) & (df.n_rounds == 1) & (~df.has_search), 'avg_time')
    m_base_cost = _mean_of(df, (df.n_agents == 1) & (df.n_rounds == 1) & (~df.has_search), 'avg_cost')

    # Multi-Agent: 3 agents, 1 round, no search
    m_ma_acc = _mean_of(df, (df.n_agents == 3) & (df.n_rounds == 1) & (~df.has_search), 'accuracy')
    m_ma_time = _mean_of(df, (df.n_agents == 3) & (df.n_rounds == 1) & (~df.has_search), 'avg_time')
    m_ma_cost = _mean_of(df, (df.n_agents == 3) & (df.n_rounds == 1) & (~df.has_search), 'avg_cost')

    # Evidence Retrieval: add search (3 agents, 1 round, with search)
    m_er_acc = _mean_of(df, (df.n_agents == 3) & (df.n_rounds == 1) & (df.has_search), 'accuracy')
    m_er_time = _mean_of(df, (df.n_agents == 3) & (df.n_rounds == 1) & (df.has_search), 'avg_time')
    m_er_cost = _mean_of(df, (df.n_agents == 3) & (df.n_rounds == 1) & (df.has_search), 'avg_cost')

    # Iterative Reasoning: add rounds (3 agents, 3 rounds, with search)
    m_ir_acc = _mean_of(df, (df.n_agents == 3) & (df.n_rounds == 3) & (df.has_search), 'accuracy')
    m_ir_time = _mean_of(df, (df.n_agents == 3) & (df.n_rounds == 3) & (df.has_search), 'avg_time')
    m_ir_cost = _mean_of(df, (df.n_agents == 3) & (df.n_rounds == 3) & (df.has_search), 'avg_cost')

    # Role Play: enable vs disable (use delta as increment)
    m_rp_enable_acc = _mean_of(role_play_df, role_play_df.exp_name == 'enable_role_play', 'accuracy')
    m_rp_disable_acc = _mean_of(role_play_df, role_play_df.exp_name == 'disable_role_play', 'accuracy')
    m_rp_enable_time = _mean_of(role_play_df, role_play_df.exp_name == 'enable_role_play', 'avg_time')
    m_rp_disable_time = _mean_of(role_play_df, role_play_df.exp_name == 'disable_role_play', 'avg_time')
    m_rp_enable_cost = _mean_of(role_play_df, role_play_df.exp_name == 'enable_role_play', 'avg_cost')
    m_rp_disable_cost = _mean_of(role_play_df, role_play_df.exp_name == 'disable_role_play', 'avg_cost')

    # Orchestration: best vs worst (use delta as increment)
    m_orch_best_acc = _mean_of(orchestration_df, orchestration_df.exp_name == 'group_chat_with_orchestrator', 'accuracy')
    m_orch_worst_acc = _mean_of(orchestration_df, orchestration_df.exp_name == 'independent', 'accuracy')
    m_orch_best_time = _mean_of(orchestration_df, orchestration_df.exp_name == 'group_chat_with_orchestrator', 'avg_time')
    m_orch_worst_time = _mean_of(orchestration_df, orchestration_df.exp_name == 'independent', 'avg_time')
    m_orch_best_cost = _mean_of(orchestration_df, orchestration_df.exp_name == 'group_chat_with_orchestrator', 'avg_cost')
    m_orch_worst_cost = _mean_of(orchestration_df, orchestration_df.exp_name == 'independent', 'avg_cost')

    # Build accuracy levels for the first 4 (cumulative path) and independent top values for RP/Orch
    acc_levels = [m_base_acc, m_ma_acc, m_er_acc, m_ir_acc, np.nan, np.nan]
    time_levels = [m_base_time, m_ma_time, m_er_time, m_ir_time, np.nan, np.nan]
    cost_levels = [m_base_cost, m_ma_cost, m_er_cost, m_ir_cost, np.nan, np.nan]

    # Deltas and bottoms for role play and orchestration (independent, not cumulative)
    rp_delta_acc = m_rp_enable_acc - m_rp_disable_acc
    rp_delta_time = m_rp_enable_time - m_rp_disable_time
    rp_delta_cost = m_rp_enable_cost - m_rp_disable_cost
    orch_delta_acc = m_orch_best_acc - m_orch_worst_acc
    orch_delta_time = m_orch_best_time - m_orch_worst_time
    orch_delta_cost = m_orch_best_cost - m_orch_worst_cost

    # For display, use top values for RP/Orch on lines
    acc_levels[4] = m_rp_enable_acc
    time_levels[4] = m_rp_enable_time
    cost_levels[4] = m_rp_enable_cost
    acc_levels[5] = m_orch_best_acc
    time_levels[5] = m_orch_best_time
    cost_levels[5] = m_orch_best_cost

    x = np.arange(len(components))
    comp_colors = colors['component_breakdown']
    bar_palette = [
        comp_colors['Baseline'],
        comp_colors['Multi-Agent'],
        comp_colors['Evidence_Retrieval'],
        comp_colors['Iterative_Reasoning'],
        comp_colors['Role_Play'],
        comp_colors['Discussion_Orchestration'],
    ]

    # Draw accuracy waterfall
    # First 4 bars are cumulative steps based on absolute levels; RP/Orch are independent deltas from their own bottoms
    running = 0.0
    for i, color in enumerate(bar_palette):
        if i == 0:
            bottom = 0.0
            height = acc_levels[0]
        elif i in (1, 2, 3):
            bottom = acc_levels[i - 1]
            height = acc_levels[i] - acc_levels[i - 1]
        elif i == 4:
            bottom = m_rp_disable_acc
            height = rp_delta_acc
        else:  # i == 5
            bottom = m_orch_worst_acc
            height = orch_delta_acc
        ax.bar(x[i], height, bottom=bottom, color=color, width=0.65, edgecolor='black', linewidth=1.0, alpha=0.9)

    # Left axis labels and styling (match 0.B)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(components, fontsize=15, rotation=0, ha='center')
    ax.tick_params(axis='both', labelsize=15)
    ax.grid(True, alpha=0.4, axis='y', linewidth=0.8, linestyle=':')
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=22, fontweight='bold')

    # Right axis: time and cost (per 100)
    ax2 = ax.twinx()
    time_color = colors['methods'].get('Few-shot', '#444444')
    cost_color = colors['methods'].get('MedRAG', '#888888')
    time_line = ax2.plot(x, time_levels, color=time_color, linestyle='-', marker='D',
                         markersize=6, linewidth=2.5, alpha=0.9,
                         label='Time (s)', markeredgecolor='white', markeredgewidth=0.8)
    cost_line = ax2.plot(x, [c * 100 for c in cost_levels], color=cost_color, linestyle='--', marker='^',
                         markersize=6, linewidth=2.5, alpha=0.9,
                         label='Cost (per 100)', markeredgecolor='white', markeredgewidth=0.8)
    ax2.set_ylabel('Time (s) / Cost (per 100)', fontweight='bold', fontsize=18)
    ax2.tick_params(axis='both', labelsize=15)

    # Loosen right-axis limits so labels always fit
    y2_values = [v for v in list(time_levels) + [vv * 100.0 for vv in cost_levels] if np.isfinite(v)]
    if y2_values:
        y2_min, y2_max = min(y2_values), max(y2_values)
        y2_pad = max(5.0, 0.15 * (y2_max - y2_min))
        ax2.set_ylim(y2_min - y2_pad, y2_max + y2_pad)

    # Legend styling: move to upper-left, no redundant title
    handles, labels = ax2.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(0.02, 0.98),
              fontsize=13, ncol=1, frameon=True, fancybox=True, shadow=True,
              framealpha=0.95, facecolor='white', edgecolor='black')

    # Limits with padding
    ymin, ymax = np.nanmin(acc_levels), np.nanmax(acc_levels)
    pad = (ymax - ymin) * 0.15 if np.isfinite(ymin) and np.isfinite(ymax) else 5
    ax.set_ylim(max(0, ymin - pad), ymax + pad)
    ax.set_xlim(-0.5, len(components) - 0.5)

    # Annotations: accuracy deltas above each bar
    for i in range(len(components)):
        if i == 0:
            top_y = acc_levels[0]
            label = f"{top_y:.1f}%"
        elif i in (1, 2, 3):
            delta = acc_levels[i] - acc_levels[i - 1]
            top_y = acc_levels[i]
            label = f"+{delta:.1f}%" if np.isfinite(delta) else ""
        elif i == 4:
            top_y = m_rp_disable_acc + max(0, rp_delta_acc)
            label = f"+{rp_delta_acc:.1f}%"
        else:  # 5
            top_y = m_orch_worst_acc + max(0, orch_delta_acc)
            label = f"+{orch_delta_acc:.1f}%"

        if label:
            ax.annotate(label, xy=(x[i], top_y), xycoords='data',
                        textcoords='offset points', xytext=(0, 6),
                        ha='center', va='bottom', fontsize=12, fontweight='bold', clip_on=True)

    return ax


DATASETS = [
    ('MedQA',        'Jin et al., 2020',      1273, 100),
    ('PubMedQA',     'Jin et al., 2019',       500, 100),
    ('MedMCQA',      'Pal et al., 2022',      2816, 100),
    ('MedBullets',   'Chen et al., 2024',      308,  89),
    ('MedExQA',      'Kim et al., 2024',       935, 100),
    ('MedXpertQA-R', 'Zuo et al., 2025',      1225, 100),
    ('MedXpertQA-U', 'Zuo et al., 2025',      1225, 100),
    ('MMLU-Med',     'Hendrycks et al., 2020', 1089,  73),
    ('MMLU-Pro-Med', 'Wang et al., 2024',      818, 100),
]


PALETTE = [
    '#4E79A7',  # steel blue
    '#F28E2B',  # warm orange
    '#E15759',  # soft red
    '#76B7B2',  # teal
    '#59A14F',  # sage green
    '#EDC948',  # gold
    '#B07AA1',  # mauve
    '#FF9DA7',  # rose
    '#9C755F',  # taupe
]


def _connector(ax, ang_deg, radius, palette_color, name, cite, original_val, hard_val,
               inner_r=1.6, outer_r=1.8, tail_len=0.3, gap=0.06):
    """Draw a single elbow connector with label, Nature Methods style."""
    rad = np.deg2rad(ang_deg)

    x0 = inner_r * np.cos(rad)
    y0 = inner_r * np.sin(rad)
    x1 = outer_r * np.cos(rad)
    y1 = outer_r * np.sin(rad)

    # Horizontal tail direction
    right_side = x1 >= 0
    x2 = x1 + (tail_len if right_side else -tail_len)
    ha = 'left' if right_side else 'right'
    sign = 1 if right_side else -1

    # Elbow line — thin, grey
    ax.plot([x0, x1, x2], [y0, y1, y1],
            color='#666666', linewidth=1.0, solid_capstyle='round', clip_on=False, alpha=0.8)

    # Tiny dot at wedge edge
    ax.plot(x0, y0, 'o', color=palette_color, markersize=4.0,
            clip_on=False, zorder=5,
            markeredgecolor='white', markeredgewidth=0.5)

    # Dataset name — medium weight, dark grey
    ax.text(x2 + sign * gap, y1 + 0.03,
            name, ha=ha, va='bottom',
            fontsize=11, fontweight='bold', color='#1a1a1a',
            clip_on=False)

    # Citation + original → hard on second line — lighter, smaller
    ax.text(x2 + sign * gap, y1 - 0.08,
            f'{cite}  ({original_val} → {hard_val})',
            ha=ha, va='top',
            fontsize=8.5, color='#666666', style='italic',
            clip_on=False)


def plot_benchmark_distribution(ax, panel_label='d'):
    """Nature Methods–style nested donut chart showing original vs filtered dataset composition."""

    names        = [d[0] for d in DATASETS]
    cites        = [d[1] for d in DATASETS]
    original_vals = [d[2] for d in DATASETS]
    hard_vals    = [d[3] for d in DATASETS]
    total_original = sum(original_vals)
    total_hard   = sum(hard_vals)

    # Create exploded effect for better separation
    explode = [0.02] * len(hard_vals)  # Small separation for all wedges

    # Outer ring: Original dataset sizes (more transparent) - made larger
    wedges_outer, _ = ax.pie(
        original_vals,
        startangle=90,
        colors=[c + '60' for c in PALETTE],  # More transparency for outer ring
        explode=explode,
        wedgeprops=dict(width=0.35, edgecolor='white', linewidth=2.0),
        radius=1.55,
    )

    # Inner ring: Filtered/hard subset sizes (solid colors) - made larger
    wedges_inner, _ = ax.pie(
        hard_vals,
        startangle=90,
        colors=PALETTE,
        explode=explode,
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2.5),
        radius=1.2,
    )

    # Add a larger subtle background circle for better text readability
    circle = plt.Circle((0, 0), 0.45, color='white', alpha=0.9, zorder=1)
    ax.add_patch(circle)
    circle_border = plt.Circle((0, 0), 0.45, fill=False, edgecolor='#e0e0e0',
                              linewidth=1.5, alpha=0.7, zorder=2)
    ax.add_patch(circle_border)

    # Center: benchmark name + sample sizes with better hierarchy and improved readability
    # Main title with stronger background - adjusted for larger circle
    ax.text(0, 0.25, 'MedAgentsBench',
            ha='center', va='center',
            fontsize=18, fontweight='bold', color='#1a1a1a',
            path_effects=[pe.withStroke(linewidth=4, foreground='white')],
            zorder=3)

    # Subtitle with better spacing and contrast
    ax.text(0, 0.08, f'Original: {total_original:,} samples',
            ha='center', va='center',
            fontsize=12, color='#555555', fontweight='medium',
            path_effects=[pe.withStroke(linewidth=3, foreground='white')],
            zorder=3)

    # Filtered count with emphasis and better positioning
    ax.text(0, -0.08, f'Filtered: {total_hard} samples',
            ha='center', va='center',
            fontsize=13, fontweight='bold', color='#2c2c2c',
            path_effects=[pe.withStroke(linewidth=3, foreground='white')],
            zorder=3)

    # Connectors - use outer wedges for positioning with adjusted parameters
    for i, (wedge, name, cite, orig_val, hard_val) in enumerate(zip(wedges_outer, names, cites, original_vals, hard_vals)):
        ang = (wedge.theta2 + wedge.theta1) / 2.0
        _connector(ax, ang, 1.55, PALETTE[i], name, cite, orig_val, hard_val)

    # Add improved legend for original vs filtered
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE[0] + '60', edgecolor='white',
                     linewidth=2.0, label='Original Dataset'),
        plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE[0], edgecolor='white',
                     linewidth=2.5, label='Filtered Subset')
    ]
    legend = ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1.05),
                      fontsize=11, frameon=True, fancybox=True, shadow=True, framealpha=0.95,
                      facecolor='white', edgecolor='#cccccc', borderpad=0.8)
    legend.get_frame().set_linewidth(1.5)

    ax.set_aspect('equal')
    ax.set_xlim(-3.0, 3.0)  # Expanded to accommodate larger circles
    ax.set_ylim(-2.7, 2.7)  # Expanded to accommodate larger circles
    ax.axis('off')

    # Panel label
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=22, fontweight='bold')

    return ax


def create_figure_0():
    """Create Figure 0: Main Comparison"""

    apply_medagents_style()

    colors = get_figure_0_colors()

    base_dir = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base_dir, 'data', 'main_comparison.csv'))
    print(f"Loaded {len(df)} records from real data")

    agent_df = pd.read_csv(os.path.join(base_dir, 'data', 'agent_configuration.csv'))
    role_play_df = pd.read_csv(os.path.join(base_dir, 'data', 'role_play.csv'))
    orchestration_df = pd.read_csv(os.path.join(base_dir, 'data', 'orchestration_style.csv'))
    print(f"Loaded agent configuration data for component breakdown")

    fig = plt.figure(figsize=(18, 22))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 1, 2], width_ratios=[1, 1], hspace=0.3, wspace=0.3)

    # a: full width top row
    ax1 = fig.add_subplot(gs[0, :])
    plot_overall_performance_bar(ax1, df, colors, panel_label='a')

    # b: left column, spanning rows 1-2
    ax2 = fig.add_subplot(gs[1:, 0])
    plot_pareto_frontier(ax2, df, colors, panel_label='b')

    # c: top-right
    ax3 = fig.add_subplot(gs[1, 1])
    plot_component_breakdown_waterfall(ax3, agent_df, role_play_df, orchestration_df, colors, panel_label='c')

    # d: bottom-right (under c)
    ax4 = fig.add_subplot(gs[2, 1])
    plot_benchmark_distribution(ax4, panel_label='d')

    plt.tight_layout()
    save_figure(fig, 'fig-2.main_comparison.pdf')

    return fig


if __name__ == "__main__":
    create_figure_0()
