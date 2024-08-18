import json

def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_diff(new_data, old_data):
    diff = {}
    for dataset in new_data:
        diff[dataset] = {}
        for model in new_data[dataset]:
            diff[dataset][model] = {}
            for metric in new_data[dataset][model]:
                diff[dataset][model][metric] = new_data[dataset][model][metric] - old_data[dataset][model][metric]
    return diff

def process_dataset_name(name):
    # Split by underscore and capitalize each word
    words = name.split('_')
    return ' '.join(word.capitalize() for word in words)

def generate_latex_table(diff_data):
    latex_table = r"""\begin{table}[ht!]
\caption{Difference in log-loss and accuracy between the new test set (2024-03-31 to 2024-06-30) and the original test set (2023-03-31 to 2024-03-31). Positive values indicate better performance on the new test set. Both sets of evaluations are run with the same set of hyperparameters, those found by the hyperparameter sweep on the original train set.}
\label{new-test-set-table}
\centering
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{ll|rrrrrrrrrrr}
\toprule
Game & Metrics & Elo & Glicko & Glicko 2 & TrueSkill & \makecell{W\&L \\ BT} & \makecell{W\&L \\ TM} & mElo & \makecell{vSKF \\ BT} & \makecell{vSKF \\ TM} & GenElo & vElo \\
\midrule
"""

    models = ["elo", "glicko", "glicko2", "trueskill", "wl_bt", "wl_tm", "melo", "vskf_bt", "vskf_tm", "genelo", "velo"]

    for dataset in diff_data:
        processed_name = process_dataset_name(dataset)
        latex_table += f"\\multirow{{2}}{{*}}{{{processed_name}}} & Accuracy"
        for model in models:
            value = diff_data[dataset].get(model, {}).get("accuracy", 0)
            latex_table += f" & {value:.4f}"
        latex_table += r" \\"
        latex_table += "\n & Log Loss"
        for model in models:
            value = diff_data[dataset].get(model, {}).get("log_loss", 0)
            latex_table += f" & {value:.4f}"
        latex_table += r" \\"
        latex_table += "\n"

    latex_table += r"""\bottomrule
\end{tabular}%
}
\end{table}
"""

    return latex_table

# Main execution
new_data = read_json('new_results.json')
old_data = read_json('original_results.json')
diff_data = calculate_diff(new_data, old_data)

latex_table = generate_latex_table(diff_data)
print(latex_table)

# Save the LaTeX table to a text file
with open('latex_table.txt', 'w') as f:
    f.write(latex_table)

# Save the diff data to a new JSON file
with open('diff_results.json', 'w') as f:
    json.dump(diff_data, f, indent=2)

print("LaTeX table has been saved to 'latex_table.txt'")
print("Difference results have been saved to 'diff_results.json'")