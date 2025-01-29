import matplotlib.pyplot as plt
import numpy as np
import re
import glob
import os


def parse_results_file(filename):
    with open(filename, 'r') as f:
        content = f.read()

    tumor_pattern = r"tumor \(Label 10\):\nMean coverage: ([\d.]+) ± ([\d.]+)"
    tumor_match = re.search(tumor_pattern, content, re.IGNORECASE)

    if tumor_match:
        mean_coverage = float(tumor_match.group(1))
        std_coverage = float(tumor_match.group(2))
        return mean_coverage, std_coverage

    return None, None


def get_confidence_from_folder(folder_name):
    match = re.search(r'(?:sacp|cp)(0\.\d+)', folder_name)
    if match:
        return float(match.group(1))
    return None


def collect_data(base_path):
    folders = glob.glob(os.path.join(base_path, "multilabel_results_*"))
    data = []

    for folder in folders:
        confidence = get_confidence_from_folder(os.path.basename(folder))
        if confidence is None:
            continue

        results_file = os.path.join(folder, "multilabel_test_results.txt")
        if not os.path.exists(results_file):
            print(f"Missing results file in {folder}")
            continue

        mean_coverage, std_coverage = parse_results_file(results_file)
        if mean_coverage is not None:
            data.append((confidence, mean_coverage, std_coverage))

    return sorted(data)


plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

weighted_data = collect_data('MSKsacp')  # Anatomically-aware CP results
unweighted_data = collect_data('MSKcp')  # Standard CP results

plt.figure(figsize=(10, 8))

ideal_line = np.linspace(0, 1, 100)
plt.plot(ideal_line, ideal_line, '--', color='gray', label='Ideal', zorder=1)


if weighted_data:
    conf_w, mean_w, std_w = zip(*weighted_data)
    plt.errorbar(conf_w, mean_w, yerr=std_w, fmt='o-', color='#4C72B0',
                 label='Anatomically-Aware CP', capsize=5, zorder=3)

if unweighted_data:
    conf_u, mean_u, std_u = zip(*unweighted_data)
    plt.errorbar(conf_u, mean_u, yerr=std_u, fmt='s-', color='#C44E52',
                 label='Standard CP', capsize=5, zorder=2)

plt.xlabel('Target Confidence Level')
plt.ylabel('Achieved Coverage')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.ylim(0, 1.0)
plt.xlim(0, 1.0)

plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))

print("\nWeighted (Anatomically-Aware) CP Results:")
for conf, mean, std in weighted_data:
    print(f"{conf:.0%} -> {mean:.1%} ± {std:.1%}")

print("\nUnweighted (Standard) CP Results:")
for conf, mean, std in unweighted_data:
    print(f"{conf:.0%} -> {mean:.1%} ± {std:.1%}")

plt.tight_layout()
plt.savefig('figure3-1-MSK.pdf', dpi=300, bbox_inches='tight')
plt.close()