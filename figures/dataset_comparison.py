"""Dataset comparison. Run with python -m figures.dataset_comparison."""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from .style import apply_medagents_style, apply_standard_plot_formatting, get_background_colors, get_figure_0_colors, save_figure


def _dataset_name_map():
    return {
        'medbullets': 'MedBullets',
        'medexqa': 'MedExQA',
        'medmcqa': 'MedMCQA',
        'medqa': 'MedQA',
        'medxpertqa-r': 'MedXpertQA-R',
        'medxpertqa-u': 'MedXpertQA-U',
        'mmlu': 'MMLU-Med',
        'mmlu-pro': 'MMLU-Pro-Med',
        'pubmedqa': 'PubMedQA',
    }


def create_figures():
    """Create Supplemental Figure S0: Dataset Comparison"""

    apply_medagents_style()
    colors = get_figure_0_colors()

    # Load experimental data
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'main_comparison.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Expected experimental data at '{data_path}' for Figure S0."
        )
    raw = pd.read_csv(data_path)
    print(f"Loaded {len(raw)} records from experimental data")

    # Create figure with grid layout and aligned width
    fig = plt.figure(figsize=(18, 22))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1], hspace=0.25)

    # Subfigure A: GraphPad Prism style bar plots (3x3)
    gs_A = gs[0, 0].subgridspec(3, 3, hspace=0.2, wspace=0.15)

    # Process data for subplot A
    method_colors = colors.get('methods', {})
    background_colors = get_background_colors()
    models = ['o3-mini', 'gpt-4o', 'gpt-4o-mini']
    methods = ['CoT', 'CoT-SC', 'MedPrompt', 'MultiPersona', 'MedAgents', 'AFlow',
               'MedAgents-X', 'Few-shot', 'MDAgents', 'SPO', 'Self-refine', 'MedRAG', 'Zero-shot']

    name_map = _dataset_name_map()
    raw['dataset'] = raw['dataset'].map(lambda x: name_map.get(x, x))
    all_datasets = sorted(raw['dataset'].dropna().unique().tolist())
    priority_datasets = ['MMLU-Med', 'MedBullets', 'MedQA']
    other_datasets = [d for d in all_datasets if d not in priority_datasets]
    datasets = priority_datasets + other_datasets[:6]
    model_alphas = [0.8, 0.6, 0.4]

    # Plot each dataset in subplot A
    for idx, dataset in enumerate(datasets):
        row, col = idx // 3, idx % 3
        ax = fig.add_subplot(gs_A[row, col])

        dataset_data = raw[raw['dataset'] == dataset]
        available_methods = [m for m in methods if m in dataset_data['method'].values]
        x_positions = np.arange(len(available_methods))

        for method_idx, method in enumerate(available_methods):
            method_data = dataset_data[dataset_data['method'] == method]
            method_color = method_colors.get(method, '#cccccc')

            accuracies = []
            std_devs = []
            for model in models:
                model_data = method_data[method_data['model'] == model]
                if len(model_data) > 0:
                    runs = model_data['accuracy'].values
                    accuracies.append(np.mean(runs))
                    std_devs.append(np.std(runs) if len(runs) > 1 else 0)
                else:
                    accuracies.append(None)
                    std_devs.append(None)

            valid_data = [(acc, std) for acc, std in zip(accuracies, std_devs) if acc is not None]
            if len(valid_data) > 0:
                valid_accuracies = [acc for acc, std in valid_data]
                baseline = min(valid_accuracies)
                x_pos = x_positions[method_idx]

                ax.bar(x_pos, baseline, 0.6, color=method_color, alpha=1,
                      edgecolor='black', linewidth=0.5)

                bottom = baseline
                for model_idx, (acc, std) in enumerate(valid_data):
                    delta = acc - baseline
                    if delta > 0:
                        ax.bar(x_pos, delta, 0.6, bottom=bottom,
                              color=method_color, alpha=model_alphas[model_idx % len(model_alphas)],
                              edgecolor='black', linewidth=0.3)
                        bottom += delta

                for model_idx, (acc, std) in enumerate(valid_data):
                    if std > 0:
                        ax.errorbar(x_pos, acc, yerr=std,
                                   fmt='none', color='black', capsize=3, capthick=1,
                                   linewidth=1, zorder=10)

        ax.set_title(dataset, fontweight='bold', fontsize=14, pad=10)

        if row == 2:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(available_methods, rotation=90, ha='center', fontweight='bold', fontsize=11)
        else:
            ax.set_xticks([])

        if col != 0:
            ax.tick_params(left=False, labelleft=False)

        ax.grid(True, alpha=0.3, linewidth=0.5, axis='y')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.set_facecolor(background_colors.get('white', '#fafafa'))
        ax.set_ylim(0, 80)

        if idx == 0:
            apply_standard_plot_formatting(ax, 'a', background_color=background_colors['white'], fontsize=22)

    # Subfigure B: Pareto frontier by models (3x1) with shared y-axis
    gs_B = gs[1, 0].subgridspec(1, 3, hspace=0.2, wspace=0.15)

    # Create axes with shared y-axis
    models_B = ['o3-mini', 'gpt-4o', 'gpt-4o-mini']
    axes_B = []
    for model_idx, model in enumerate(models_B):
        if model_idx == 0:
            ax = fig.add_subplot(gs_B[0, model_idx])
            axes_B.append(ax)
        else:
            ax = fig.add_subplot(gs_B[0, model_idx], sharey=axes_B[0])
            axes_B.append(ax)

    for model_idx, model in enumerate(models_B):
        ax = axes_B[model_idx]

        # Filter data for this model
        model_data = raw[raw['model'] == model]

        # Calculate average metrics per method for this model
        if len(model_data) > 0:
            avg_metrics = model_data.groupby('method').agg({
                'accuracy': 'mean',
                'avg_cost': lambda x: x.mean() * 100,  # Convert to cents
                'avg_time': 'mean'
            }).reset_index()

            # Plot scatter points
            for _, row in avg_metrics.iterrows():
                if row['method'] == 'MedAgents-X':
                    ax.scatter(row['avg_cost'], row['accuracy'],
                              c=method_colors.get(row['method'], 'gray'),
                              s=400, alpha=0.9,
                              edgecolors='black', linewidth=2.5,
                              marker='*', zorder=10)
                else:
                    ax.scatter(row['avg_cost'], row['accuracy'],
                              c=method_colors.get(row['method'], 'gray'),
                              s=180, alpha=0.8,
                              edgecolors='black', linewidth=1.5,
                              zorder=5)

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
                ax.plot(pareto_data['avg_cost'], pareto_data['accuracy'], 'k--', alpha=0.8, linewidth=2.5,
                       zorder=8)

            # Add method annotations for key methods
            key_methods = ['MedAgents-X', 'CoT', 'Few-shot', 'Zero-shot', 'MedRAG']
            for _, row in avg_metrics.iterrows():
                if row['method'] in key_methods:
                    offset_x = 8 if row['method'] != 'MedAgents-X' else 10
                    offset_y = 6 if row['method'] != 'MedAgents-X' else 8
                    ax.annotate(row['method'], (row['avg_cost'], row['accuracy']),
                               xytext=(offset_x, offset_y), textcoords='offset points',
                               fontsize=12, ha='left', va='bottom', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9,
                                        edgecolor='gray', linewidth=1))

        # Styling
        ax.set_title(model, fontweight='bold', fontsize=16, pad=15)
        ax.set_xlabel('Average Cost (cents per query)', fontweight='bold', fontsize=14)

        # Only show y-axis label and ticks on leftmost subplot
        if model_idx == 0:
            ax.set_ylabel('Average Accuracy (%)', fontweight='bold', fontsize=14)
        else:
            ax.tick_params(left=False, labelleft=False)

        # Set appropriate x-limits based on model
        if model == 'gpt-4o-mini':
            ax.set_xlim(-0.1, 2.0)
        elif model == 'gpt-4o':
            ax.set_xlim(-0.5, 12.0)
        elif model == 'o3-mini':
            ax.set_xlim(-0.5, 10.0)

        ax.set_ylim(10, 45)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
        ax.set_facecolor(background_colors.get('white', '#fafafa'))

        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        # Apply panel label to first subplot
        if model_idx == 0:
            apply_standard_plot_formatting(ax, 'b', background_color=background_colors['white'], fontsize=22)

    # Add Pareto frontier legend to the last subplot
    pareto_legend = [plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2.5,
                               label='Pareto Frontier', alpha=0.8)]
    axes_B[-1].legend(handles=pareto_legend, loc='lower right', fontsize=12,
                     frameon=True, fancybox=True, shadow=True, framealpha=0.95,
                     facecolor='white', edgecolor='black')

    # Save figure
    plt.tight_layout()
    save_figure(fig, 'fig-s0.dataset_comparison.pdf')

    return fig


if __name__ == "__main__":
    create_figures()
