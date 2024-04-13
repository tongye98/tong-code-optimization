import pandas as pd

result_pth = "/data/tongye/saves/codellama_13b_sft_pie_cpp_0123/result_1sample_temp1_topp1_report1000.json"
run_metrics = pd.read_json(result_pth,lines=True,orient="records")
len(run_metrics)

gen_col = "candidates"
correct_run_metrics = run_metrics[(run_metrics[f"{gen_col}_acc"] > 0.99) & (run_metrics["input_acc"] > 0.99)]
len(correct_run_metrics)

def mean_std(df, col) -> str:
    mean_col = f"{col}_mean"
    std_col = f"{col}_std"
    if mean_col not in df.columns or std_col not in df.columns:
        return f"{df[col].mean():.4f} ± {df[col].std():.4f}"

    return f"{df[mean_col].mean():.4f} ± {df[std_col].mean():.4f}"

print("---Execution time---")
print(
    f"[Reported in CodeNet] input program (ms): {mean_std(correct_run_metrics, 'cpu_time_v0')}"
)
print(
    f"[Reported in CodeNet] reference (output) program (ms): {mean_std(correct_run_metrics, 'cpu_time_v1')}"
)

print("-" * 80)
print(f"[Our measurement] input program (ms): {mean_std(correct_run_metrics, 'input_time')}")
print(
    f"[Our measurement] reference (output) program (ms): {mean_std(correct_run_metrics, 'reference_time')}"
)
print(
    f"[Our measurement] {gen_col} program (ms): {mean_std(correct_run_metrics, f'{gen_col}_time')}"
)


# run_metrics_improved = run_metrics[run_metrics[f"{gen_col}_time_mean"] < run_metrics["input_time_mean"]]
from math import sqrt
def t_test(df,gen_col,n):
    mean_model = df[f'{gen_col}_time_mean']
    mean_input = df['input_time_mean']
    std_dev_model = df[f'{gen_col}_time_std']
    std_dev_input = df['input_time_std']

    t = (mean_input - mean_model) / ((std_dev_model**2/n) + (std_dev_input**2/n)).apply(sqrt)
    return t
    
improve_frac = 0.0
# run_metrics_improved = correct_run_metrics[((correct_run_metrics["input_time_mean"] - correct_run_metrics[f"{gen_col}_time_mean"]) / correct_run_metrics["input_time_mean"]) > improve_frac]
run_metrics_improved = correct_run_metrics[t_test(correct_run_metrics,gen_col,25) > 0.05]

len(run_metrics_improved)

print("----Metrics when improved--")
print(
    f"Found {len(correct_run_metrics)}/{len(run_metrics)}={len(correct_run_metrics)/len(run_metrics)*100}% problems where the {gen_col} program is still correct"
)
print(
    f"Found {len(run_metrics_improved)}/{len(run_metrics)}={len(run_metrics_improved)/len(run_metrics)*100}% problems where the {gen_col} program is faster than the input program"
)
print(
    f"[Our measurement] input program (ms): {mean_std(run_metrics_improved, 'input_time')}"
)
print(
    f"[Our measurement] reference (output) program (ms): {mean_std(run_metrics_improved, 'reference_time')}"
)
print(
    f"[Our measurement] {gen_col} program (ms): {mean_std(run_metrics_improved, f'{gen_col}_time')}"
)
print(
    f"[Our measurement] Average SpeedUp: {(run_metrics_improved['input_time_mean'] / run_metrics_improved[f'{gen_col}_time_mean']).mean()}"
)
print(
    f"[Our measurement] Relative Time Reduction: {((run_metrics_improved['input_time_mean'] - run_metrics_improved[f'{gen_col}_time_mean'])/run_metrics_improved['input_time_mean']).mean()}"
)

def get_anomalies(run_metrics):
    run_metrics["codenet_reported_rel_improvement"] = (
        run_metrics["cpu_time_v0"] - run_metrics["cpu_time_v1"]
    ) / run_metrics["cpu_time_v0"]
    run_metrics["codenet_reported_rel_improvement"] = run_metrics[
        "codenet_reported_rel_improvement"
    ].apply(lambda x: round(x * 100, 2))
    run_metrics["measured_rel_improvement"] = (
        run_metrics["input_time_mean"] - run_metrics["reference_time_mean"]
    ) / run_metrics["input_time_mean"]
    run_metrics["measured_rel_improvement"] = run_metrics["measured_rel_improvement"].apply(
        lambda x: round(x * 100, 2)
    )
    run_metrics["is_anomaly"] = run_metrics.apply(
        lambda x: x["codenet_reported_rel_improvement"] > 10 and x["measured_rel_improvement"] < 0,
        axis=1,
    )
    run_metrics_anomalies = run_metrics[run_metrics["is_anomaly"]]
    return run_metrics_anomalies

print(f"Number of cases where reference took longer by our measurement: {len(get_anomalies(run_metrics))}")