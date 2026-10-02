"""Tables. Run with python -m figures.tables."""
import numpy as np
import pandas as pd
from pathlib import Path


def format_result(mean, std, bold=False, underline=False):
    """Format result with mean ± std, bold if specified, underline if second best"""
    if pd.isna(mean) or pd.isna(std):
        return "--"

    formatted = f"{mean:.1f} ± {{\\scriptsize {std:.1f}}}"
    if bold:
        formatted = f"\\textbf{{{formatted}}}"
    elif underline:
        formatted = f"\\underline{{{formatted}}}"
    return formatted


def generate_main_comparison_table():
    """Generate main comparison table across all methods and datasets with different base models"""

    df = pd.read_csv(Path(__file__).parent / 'data/main_comparison.csv')

    summary = df.groupby(['method', 'dataset', 'model']).agg({
        'accuracy': ['mean', 'std'],
        'avg_time': ['mean', 'std'],
        'avg_cost': ['mean', 'std']
    }).round(1)

    summary.columns = ['acc_mean', 'acc_std', 'time_mean', 'time_std', 'cost_mean', 'cost_std']
    summary = summary.reset_index()

    datasets = ['medqa', 'medmcqa', 'pubmedqa', 'mmlu', 'medbullets', 'medexqa', 'medxpertqa-r', 'medxpertqa-u', 'mmlu-pro']
    dataset_names = {
        'medqa': 'MedQA',
        'medmcqa': 'MedMCQA',
        'pubmedqa': 'PubMedQA',
        'mmlu': 'MMLU-Med',
        'medbullets': 'MedBullets',
        'medexqa': 'MedExQA',
        'medxpertqa-r': 'MedXpert-R',
        'medxpertqa-u': 'MedXpert-U',
        'mmlu-pro': 'MMLU-Pro'
    }

    methods = ['Zero-shot', 'Few-shot', 'CoT', 'CoT-SC', 'Self-refine', 'MedPrompt',
               'MedRAG', 'MultiPersona', 'MDAgents', 'AFlow', 'SPO', 'MedAgents', 'MedAgents-X']

    models = ['gpt-4o', 'gpt-4o-mini', 'o3-mini']
    model_names = {
        'gpt-4o': 'GPT-4o',
        'gpt-4o-mini': 'GPT-4o-mini',
        'o3-mini': 'o3-mini'
    }

    latex = []
    latex.append("\\begin{table*}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Main Results: Accuracy (\\%) comparison across medical QA datasets and base models. Best results are \\textbf{bolded}, second best are \\underline{underlined}. Results show mean ± std over 3 runs.}")
    latex.append("\\label{tab:main_results}")
    latex.append("\\resizebox{\\textwidth}{!}{")
    latex.append("\\begin{tabular}{l" + "c" * len(datasets) + "c}")
    latex.append("\\toprule")

    header = "Method & " + " & ".join([dataset_names[d] for d in datasets]) + " & Average \\\\"
    latex.append(header)
    latex.append("\\midrule")

    for model_idx, model in enumerate(models):
        latex.append(f"\\multicolumn{{{len(datasets) + 2}}}{{c}}{{\\textbf{{{model_names[model]}}}}} \\\\")
        latex.append("\\midrule")

        best_results = {}
        second_best_results = {}
        best_avg = None
        second_best_avg = None

        for dataset in datasets:
            dataset_model_data = summary[(summary['dataset'] == dataset) & (summary['model'] == model)]
            if not dataset_model_data.empty:
                sorted_data = dataset_model_data.sort_values('acc_mean', ascending=False)
                if len(sorted_data) >= 1:
                    best_results[dataset] = sorted_data.iloc[0]['acc_mean']
                if len(sorted_data) >= 2:
                    second_best_results[dataset] = sorted_data.iloc[1]['acc_mean']

        method_averages = []
        for method in methods:
            method_model_data = summary[(summary['method'] == method) & (summary['model'] == model)]
            if not method_model_data.empty:
                valid_results = []
                for dataset in datasets:
                    method_dataset_data = method_model_data[method_model_data['dataset'] == dataset]
                    if not method_dataset_data.empty:
                        valid_results.append(method_dataset_data.iloc[0]['acc_mean'])
                if valid_results:
                    avg_acc = np.mean(valid_results)
                    method_averages.append((method, avg_acc))

        if method_averages:
            method_averages.sort(key=lambda x: x[1], reverse=True)
            if len(method_averages) >= 1:
                best_avg = method_averages[0][1]
            if len(method_averages) >= 2:
                second_best_avg = method_averages[1][1]

        for method in methods:
            method_model_data = summary[(summary['method'] == method) & (summary['model'] == model)]
            if method_model_data.empty:
                continue

            row = [method]
            valid_results = []

            for dataset in datasets:
                method_dataset_data = method_model_data[method_model_data['dataset'] == dataset]
                if method_dataset_data.empty:
                    row.append("--")
                else:
                    mean_acc = method_dataset_data.iloc[0]['acc_mean']
                    std_acc = method_dataset_data.iloc[0]['acc_std']
                    valid_results.append(mean_acc)

                    is_best = dataset in best_results and abs(mean_acc - best_results[dataset]) < 0.1
                    is_second_best = (dataset in second_best_results and
                                    abs(mean_acc - second_best_results[dataset]) < 0.1 and
                                    not is_best)

                    row.append(format_result(mean_acc, std_acc, bold=is_best, underline=is_second_best))

            if valid_results:
                avg_acc = np.mean(valid_results)
                avg_std = np.std(valid_results)

                is_best_avg = best_avg is not None and abs(avg_acc - best_avg) < 0.1
                is_second_best_avg = (second_best_avg is not None and
                                     abs(avg_acc - second_best_avg) < 0.1 and
                                     not is_best_avg)

                row.append(format_result(avg_acc, avg_std, bold=is_best_avg, underline=is_second_best_avg))
            else:
                row.append("--")

            latex.append(" & ".join(row) + " \\\\")

        if model_idx < len(models) - 1:
            latex.append("\\midrule")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}}")
    latex.append("\\end{table*}")

    return "\n".join(latex)


def main():
    """Generate main LaTeX table"""

    print("Generating ICML-style LaTeX table...")

    main_table = generate_main_comparison_table()
    output = Path(__file__).resolve().parents[1] / 'output/figures'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'tab-s3.main_comparison.tex').open('w') as f:
        f.write(main_table)
    print("Generated: tab-s3.main_comparison.tex")

    print("\nDone! Main table generated in ICML style.")


if __name__ == "__main__":
    main()
