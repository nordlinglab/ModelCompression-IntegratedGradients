import numpy as np
from scipy import stats as scipy_stats

# Input data
Student = np.array([91.19, 90.5, 91.32, 91.28, 91.36, 91.35, 91.43, 91.05, 90.86, 91.5])
KD_IG = np.array([92.58, 91.73, 92.08, 92.45, 91.96, 91.58, 91.82, 91.62, 92.13, 92.23])
KD = np.array([91.73, 91.45, 91.19, 91.37, 92.29, 91.88, 92.01, 91.41, 92.01, 90.88])
KD_AT = np.array([91.48, 92.05, 91.49, 91.69, 91.58, 91.92, 92.2, 91.62, 92.06, 91.39])
IG_AT = np.array([91.7, 91.61, 91.05, 91.84, 91.38, 91.22, 91.3, 91.5, 91.57, 91.48])
KD_IG_AT = np.array(
    [91.83, 91.51, 91.56, 91.76, 91.73, 92.42, 91.84, 91.73, 91.97, 92.2]
)
IG = np.array([91.27, 92.01, 91.22, 91.42, 91.11, 91.58, 91.09, 91.57, 91.38, 90.85])
AT = np.array([90.62, 91.26, 91.1, 90.8, 91.2, 91.58, 91.43, 91.42, 91.4, 90.82])

# Create a dictionary for easier processing
methods = {
    "Student": Student,
    "KD & IG": KD_IG,
    "KD": KD,
    "KD & AT": KD_AT,
    "IG & AT": IG_AT,
    "KD & IG & AT": KD_IG_AT,
    "IG": IG,
    "AT": AT,
}

# Calculate highest accuracy, mean, and standard deviation for each method
results = {}
for name, data in methods.items():
    results[name] = {
        "min": np.min(data),
        "max": np.max(data),
        "mean": np.mean(data),
        "std": np.std(data, ddof=1),  # ddof=1 for sample standard deviation
    }

# Print basic statistics
print("Method\t\tMin\tMean\tMax\tStd Dev")
print("-" * 50)
for name, stats_data in results.items():
    print(
        f"{name:<12}\t{stats_data['min']:.2f}\t{stats_data['mean']:.2f}\t{stats_data['max']:.2f}\t{stats_data['std']:.4f}"
    )

# Calculate paired t-tests (using Student as reference)
reference = "Student"
print("\nT-statistics and p-values (compared to Student):")
print("Delta accuracy is relative to the Teacher Acc: 93.91%")
print("-" * 50)
print("Method\t\tDelta Acc\tt-statistic\tp-value")
print("-" * 50)
student_delta = np.mean(Student) - 93.91
print(f"{reference:<12}\t{student_delta:.2f}\t\t - \t\t - ")

for name, data in methods.items():
    if name != reference:
        # Calculate paired t-test
        t_stat, p_val = scipy_stats.ttest_rel(data, methods[reference])
        delta = np.mean(data) - 93.91
        print(f"{name:<12}\t{delta:+.2f}%\t\t{t_stat:.4f}\t\t{p_val:.6f}")

# Calculate t-tests between all pairs
print("\nPairwise t-tests between all methods:")
print("-" * 70)
print("Method A\t\tMethod B\t\tDelta\t\tt-statistic\tp-value")
print("-" * 70)

for name_a in methods.keys():
    for name_b in methods.keys():
        if name_a < name_b:  # Avoid duplicates
            delta = np.mean(methods[name_a]) - np.mean(methods[name_b])
            t_stat, p_val = scipy_stats.ttest_rel(methods[name_a], methods[name_b])
            print(
                f"{name_a:<12}\t{name_b:<12}\t{delta:+.2f}%\t\t{t_stat:.4f}\t\t{p_val:.6f}"
            )
