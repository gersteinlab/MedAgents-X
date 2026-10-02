"""Dataset beeswarm. Run with python -m figures.dataset_beeswarm."""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from .style import apply_medagents_style, get_figure_0_colors, save_figure


def plot_dataset_beeswarm(ax, df, colors, panel_label):
    """Create dataset performance score plot averaged over base models"""

    DATASET_MAPPING = {
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

    MODEL_MAPPING = {
        'o3-mini': 'o3-mini',
        'gpt-4o': 'GPT-4o',
        'gpt-4o-mini': 'GPT-4o-mini'
    }

    BASE_MODELS = ['o3-mini', 'gpt-4o', 'gpt-4o-mini']
    METHODS = ['CoT', 'CoT-SC', 'MedPrompt', 'MultiPersona', 'MedAgents', 'AFlow',
               'MedAgents-X', 'Few-shot', 'MDAgents', 'SPO', 'Self-refine', 'MedRAG', 'Zero-shot']

    method_colors = colors.get('methods', {})

    df['dataset'] = df['dataset'].map(DATASET_MAPPING)

    if 'model' not in df.columns:
        df['model'] = 'gpt-4o'

    dataset_model_scores = {}
    for _, row in df.iterrows():
        key = f"{row['dataset']}({row['model']})"
        if key not in dataset_model_scores:
            dataset_model_scores[key] = []
        dataset_model_scores[key].append(row['accuracy'])

    overall_avg_scores = []
    for key, scores in dataset_model_scores.items():
        avg_score = np.mean(scores)
        overall_avg_scores.append((key, avg_score))

    overall_avg_scores.sort(key=lambda x: x[1], reverse=True)
    sorted_keys = [item[0] for item in overall_avg_scores]

    y_position = 0
    y_ticks = []
    y_labels = []

    for key in sorted_keys:
        dataset_name = key.split('(')[0]
        model_name = key.split('(')[1].rstrip(')')

        dataset_model_data = df[(df['dataset'] == dataset_name) & (df['model'] == model_name)]

        scores = []
        method_list = []
        for method in METHODS:
            method_data = dataset_model_data[dataset_model_data['method'] == method]
            if not method_data.empty:
                scores.append(method_data['accuracy'].mean())
                method_list.append(method)

        if scores:
            strip_width = 0.3
            n_points = len(scores)
            if n_points > 1:
                x_positions = np.linspace(-strip_width/2, strip_width/2, n_points)
            else:
                x_positions = [0]

            for j, (x_offset, score, method) in enumerate(zip(x_positions, scores, method_list)):
                color = method_colors.get(method, 'gray')
                size = 400 if method == 'MedAgents-X' else 100

                if method == 'MedAgents-X':
                    ax.scatter(score, y_position + x_offset, c=color, s=size, alpha=0.9,
                              edgecolors='black', linewidth=3, marker='*', zorder=10)
                else:
                    ax.scatter(score, y_position + x_offset, c=color, s=size, alpha=0.8,
                              edgecolors='black', linewidth=1.5, zorder=5)

        y_ticks.append(y_position)
        formatted_model = MODEL_MAPPING.get(model_name, model_name)
        y_labels.append(f"{dataset_name}\n({formatted_model})")
        y_position += 1

    ax.set_ylim(-0.5, y_position - 0.5)
    ax.set_xlim(0, 65)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=6, ha='right')
    ax.set_ylabel('Medical Datasets by Base Model', fontweight='bold', fontsize=18)
    ax.set_xlabel('Accuracy Score (%)', fontweight='bold', fontsize=18)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.3, axis='x', linewidth=0.8, linestyle=':')

    legend_elements = []
    for method in METHODS:
        if method in method_colors:
            if method == 'MedAgents-X':
                legend_elements.append(plt.scatter([], [], c=method_colors[method], s=200,
                                                 alpha=0.9, edgecolors='black', linewidth=3,
                                                 marker='*', label=f'{method} (Ours)'))
            else:
                legend_elements.append(plt.scatter([], [], c=method_colors[method], s=100,
                                                 alpha=0.8, edgecolors='black', linewidth=1.5,
                                                 label=method))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=11,
             frameon=True, fancybox=True, shadow=True, framealpha=0.95,
             facecolor='white', edgecolor='black')

    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=22, fontweight='bold')

    return ax


def create_figure_s3():
    """Create Supplemental Figure S3: Dataset Beeswarm"""

    apply_medagents_style()
    colors = get_figure_0_colors()

    base_dir = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base_dir, 'data', 'main_comparison.csv'))
    print(f"Loaded {len(df)} records from real data")

    fig, ax = plt.subplots(figsize=(11, 12))
    plot_dataset_beeswarm(ax, df, colors, panel_label='a')

    plt.tight_layout()
    save_figure(fig, 'fig-s3.dataset_beeswarm.pdf')

    return fig


if __name__ == "__main__":
    create_figure_s3()
