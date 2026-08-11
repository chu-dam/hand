#!/usr/bin/env python3

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd


# =====================================================
# Setting
# =====================================================
LOG_DIR = str(Path(__file__).resolve().parents[1] / "friction_logs" / "right")

OUT_TRIAL_CSV = os.path.join(LOG_DIR, "friction_trial_table.csv")
OUT_USED_TRIAL_CSV = os.path.join(LOG_DIR, "friction_fit_used_trials.csv")
OUT_EXCLUDED_TRIAL_CSV = os.path.join(LOG_DIR, "friction_fit_excluded_trials.csv")
OUT_SUMMARY_CSV = os.path.join(LOG_DIR, "friction_fit_summary.csv")
OUT_PARAMS_PY = os.path.join(LOG_DIR, "hand_friction_params.py")


# =====================================================
# Filtering setting
# =====================================================
# 너무 느리거나 거의 안 움직인 trial 제외
MIN_ABS_QDOT = 0.005       # rad/s
MIN_ABS_Q_DELTA = 0.02     # rad

# limit까지 간 trial만 피팅에 사용
USE_ONLY_LIMIT_REACHED = True
VALID_STOP_REASONS = ["upper_limit", "lower_limit"]

# qdot 방향과 test_effort 방향이 반대면 제외
CHECK_DIRECTION_MATCH = True

# qdot 계산 방식
# "trial_avg": q_delta / duration 사용. 추천.
# "raw_mean": CSV qdot 평균 사용. 노이즈 많으면 비추천.
QDOT_METHOD = "trial_avg"


JOINT_NAMES = [
    "joint_1_1", "joint_1_2", "joint_1_3", "joint_1_4",
    "joint_2_1", "joint_2_2", "joint_2_3", "joint_2_4",
    "joint_3_1", "joint_3_2", "joint_3_3", "joint_3_4",
    "joint_4_1", "joint_4_2", "joint_4_3", "joint_4_4",
    "joint_5_1", "joint_5_2", "joint_5_3", "joint_5_4",
]


def load_all_csv(log_dir):
    paths = sorted(glob.glob(os.path.join(log_dir, "hand_friction_sweep_joint*.csv")))

    if not paths:
        raise RuntimeError(f"No CSV files found in: {log_dir}")

    dfs = []

    for path in paths:
        try:
            df = pd.read_csv(path)
            df["source_file"] = os.path.basename(path)
            dfs.append(df)
            print(f"[LOAD] {path} | rows={len(df)}")
        except Exception as e:
            print(f"[SKIP] {path} | {e}")

    if not dfs:
        raise RuntimeError("No valid CSV loaded.")

    return pd.concat(dfs, ignore_index=True)


def get_last_stop_reason(g):
    if "stop_reason" not in g.columns:
        return ""

    s = g["stop_reason"].fillna("").astype(str)
    s = s[s != ""]

    if len(s) == 0:
        return ""

    return s.iloc[-1]


def make_trial_table(df):
    required = [
        "trial_id",
        "joint_index",
        "joint_name",
        "direction",
        "test_effort",
        "time",
        "sample_time",
        "q",
        "qdot",
        "q_start",
        "q_delta",
        "stop_reason",
        "source_file",
    ]

    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"Missing column: {col}")

    rows = []

    group_cols = ["source_file", "trial_id"]

    for (source_file, trial_id), g in df.groupby(group_cols):
        g = g.sort_values("sample_time")

        if len(g) < 5:
            continue

        joint_index = int(g["joint_index"].iloc[0])
        joint_name = str(g["joint_name"].iloc[0])
        direction = int(g["direction"].iloc[0])
        test_effort = float(g["test_effort"].iloc[0])
        effort_mag = abs(test_effort)

        t0 = float(g["sample_time"].iloc[0])
        t1 = float(g["sample_time"].iloc[-1])
        duration = t1 - t0

        if duration <= 1e-6:
            continue

        q_start = float(g["q"].iloc[0])
        q_end = float(g["q"].iloc[-1])
        q_delta = q_end - q_start

        if QDOT_METHOD == "trial_avg":
            qdot_fit = q_delta / duration

        elif QDOT_METHOD == "raw_mean":
            n = len(g)
            i0 = int(0.2 * n)
            i1 = int(0.8 * n)

            if i1 <= i0:
                continue

            qdot_fit = float(g["qdot"].iloc[i0:i1].mean())

        else:
            raise RuntimeError(f"Unknown QDOT_METHOD: {QDOT_METHOD}")

        stop_reason = get_last_stop_reason(g)

        gravity_mean = float(g["gravity_effort"].mean()) if "gravity_effort" in g.columns else np.nan
        cmd_mean = float(g["command_effort"].mean()) if "command_effort" in g.columns else np.nan

        rows.append({
            "source_file": source_file,
            "trial_id": trial_id,
            "joint_index": joint_index,
            "joint_name": joint_name,
            "direction": direction,
            "effort_mag": effort_mag,
            "test_effort": test_effort,
            "duration": duration,
            "q_start": q_start,
            "q_end": q_end,
            "q_delta": q_delta,
            "qdot_fit": qdot_fit,
            "stop_reason": stop_reason,
            "gravity_mean": gravity_mean,
            "command_mean": cmd_mean,
            "n_samples": len(g),
        })

    return pd.DataFrame(rows)


def filter_trials_for_fit(trials):
    fit_df = trials.copy()
    fit_df["exclude_reason"] = ""

    # qdot 너무 작음
    mask = np.abs(fit_df["qdot_fit"]) < MIN_ABS_QDOT
    fit_df.loc[mask, "exclude_reason"] += "small_qdot;"

    # q_delta 너무 작음
    mask = np.abs(fit_df["q_delta"]) < MIN_ABS_Q_DELTA
    fit_df.loc[mask, "exclude_reason"] += "small_q_delta;"

    # stop reason 필터
    if USE_ONLY_LIMIT_REACHED:
        mask = ~fit_df["stop_reason"].isin(VALID_STOP_REASONS)
        fit_df.loc[mask, "exclude_reason"] += "invalid_stop_reason;"

    # 방향 불일치 필터
    if CHECK_DIRECTION_MATCH:
        # test_effort와 qdot_fit의 부호가 같아야 함
        mask = np.sign(fit_df["test_effort"]) * np.sign(fit_df["qdot_fit"]) <= 0
        fit_df.loc[mask, "exclude_reason"] += "direction_mismatch;"

    used = fit_df[fit_df["exclude_reason"] == ""].copy()
    excluded = fit_df[fit_df["exclude_reason"] != ""].copy()

    return used, excluded


def fit_joint_model(trials):
    """
    Fit:
        tau = Fc * sign(qdot) + B * qdot + bias
    """
    fit_df, excluded = filter_trials_for_fit(trials)

    if len(fit_df) < 4:
        return None, fit_df, excluded

    qdot = fit_df["qdot_fit"].to_numpy(dtype=np.float64)
    tau = fit_df["test_effort"].to_numpy(dtype=np.float64)

    A = np.column_stack([
        np.sign(qdot),
        qdot,
        np.ones_like(qdot),
    ])

    param, residuals, rank, s = np.linalg.lstsq(A, tau, rcond=None)

    Fc, B, bias = param

    tau_hat = A @ param
    err = tau - tau_hat

    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((tau - np.mean(tau)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

    result = {
        "Fc": Fc,
        "B": B,
        "bias": bias,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(r2),
        "n_trials_used": len(fit_df),
        "n_trials_excluded": len(excluded),
        "qdot_min": float(np.min(qdot)),
        "qdot_max": float(np.max(qdot)),
        "tau_min": float(np.min(tau)),
        "tau_max": float(np.max(tau)),
    }

    fit_df["tau_hat"] = tau_hat
    fit_df["fit_error"] = err

    return result, fit_df, excluded


def fit_directional_lines(trials):
    """
    참고용:
      + direction: tau = a_pos + B_pos*qdot
      - direction: tau = a_neg + B_neg*qdot
    """
    used, _ = filter_trials_for_fit(trials)

    out = {}

    for label, sign in [("pos", 1), ("neg", -1)]:
        g = used[used["direction"] == sign].copy()

        if len(g) < 2:
            out[f"intercept_{label}"] = np.nan
            out[f"B_{label}"] = np.nan
            continue

        qdot = g["qdot_fit"].to_numpy(dtype=np.float64)
        tau = g["test_effort"].to_numpy(dtype=np.float64)

        A = np.column_stack([qdot, np.ones_like(qdot)])
        B_dir, intercept = np.linalg.lstsq(A, tau, rcond=None)[0]

        out[f"intercept_{label}"] = intercept
        out[f"B_{label}"] = B_dir

    return out


def save_params_py(summary):
    Fc_arr = np.zeros(20)
    B_arr = np.zeros(20)
    bias_arr = np.zeros(20)

    for _, row in summary.iterrows():
        idx = int(row["joint_index"])
        Fc_arr[idx] = float(row["Fc"])
        B_arr[idx] = float(row["B"])
        bias_arr[idx] = float(row["bias"])

    with open(OUT_PARAMS_PY, "w") as f:
        f.write("# Auto-generated friction parameters\n")
        f.write("# Model: tau_fric = Fc * tanh(TANH_K * qdot) + B * qdot + bias\n")
        f.write("# qdot unit: rad/s\n")
        f.write("import numpy as np\n\n")

        f.write("TANH_K = 20.0\n\n")

        f.write("FRIC_FC = np.array([\n")
        for v in Fc_arr:
            f.write(f"    {v:.8f},\n")
        f.write("], dtype=np.float64)\n\n")

        f.write("FRIC_B = np.array([\n")
        for v in B_arr:
            f.write(f"    {v:.8f},\n")
        f.write("], dtype=np.float64)\n\n")

        f.write("FRIC_BIAS = np.array([\n")
        for v in bias_arr:
            f.write(f"    {v:.8f},\n")
        f.write("], dtype=np.float64)\n\n")

        f.write("def compute_friction(qdot, use_bias=False, scale=1.0):\n")
        f.write("    qdot = np.asarray(qdot, dtype=np.float64)\n")
        f.write("    tau = FRIC_FC * np.tanh(TANH_K * qdot) + FRIC_B * qdot\n")
        f.write("    if use_bias:\n")
        f.write("        tau = tau + FRIC_BIAS\n")
        f.write("    return scale * tau\n")


def main():
    df = load_all_csv(LOG_DIR)
    trials = make_trial_table(df)

    trials.to_csv(OUT_TRIAL_CSV, index=False)
    print(f"\n[SAVE] trial table: {OUT_TRIAL_CSV}")

    results = []
    all_used = []
    all_excluded = []

    for joint_index, g in trials.groupby("joint_index"):
        joint_index = int(joint_index)
        joint_name = (
            JOINT_NAMES[joint_index]
            if 0 <= joint_index < len(JOINT_NAMES)
            else str(g["joint_name"].iloc[0])
        )

        fit_result, used, excluded = fit_joint_model(g)
        dir_result = fit_directional_lines(g)

        if len(used) > 0:
            all_used.append(used)

        if len(excluded) > 0:
            all_excluded.append(excluded)

        if fit_result is None:
            print()
            print("====================================")
            print(f"[JOINT] {joint_index:02d} {joint_name}")
            print("------------------------------------")
            print("[WARN] not enough valid data")
            print(f"total    = {len(g)}")
            print(f"used     = {len(used)}")
            print(f"excluded = {len(excluded)}")
            continue

        row = {
            "joint_index": joint_index,
            "joint_name": joint_name,
            **fit_result,
            **dir_result,
            "n_trials_total": len(g),
            "n_files": g["source_file"].nunique(),
        }

        results.append(row)

        print()
        print("====================================")
        print(f"[JOINT] {joint_index:02d} {joint_name}")
        print("------------------------------------")
        print(f"Fc   = {fit_result['Fc']:.6f}")
        print(f"B    = {fit_result['B']:.6f}")
        print(f"bias = {fit_result['bias']:.6f}")
        print(f"rmse = {fit_result['rmse']:.6f}")
        print(f"R^2  = {fit_result['r2']:.4f}")
        print(f"used = {fit_result['n_trials_used']} / {len(g)}")
        print(f"excluded = {fit_result['n_trials_excluded']}")
        print(f"qdot range = {fit_result['qdot_min']:.4f} ~ {fit_result['qdot_max']:.4f}")
        print(f"tau range  = {fit_result['tau_min']:.4f} ~ {fit_result['tau_max']:.4f}")

    if all_used:
        used_df = pd.concat(all_used, ignore_index=True)
        used_df.to_csv(OUT_USED_TRIAL_CSV, index=False)
        print(f"\n[SAVE] used trials: {OUT_USED_TRIAL_CSV}")
    else:
        print("\n[WARN] no used trials")

    if all_excluded:
        excluded_df = pd.concat(all_excluded, ignore_index=True)
        excluded_df.to_csv(OUT_EXCLUDED_TRIAL_CSV, index=False)
        print(f"[SAVE] excluded trials: {OUT_EXCLUDED_TRIAL_CSV}")
    else:
        print("[INFO] no excluded trials")

    if not results:
        raise RuntimeError("No joint fitted.")

    summary = pd.DataFrame(results).sort_values("joint_index")
    summary.to_csv(OUT_SUMMARY_CSV, index=False)

    print()
    print("[SAVE] summary:", OUT_SUMMARY_CSV)

    save_params_py(summary)
    print("[SAVE] params:", OUT_PARAMS_PY)

    print()
    print("[DONE]")


if __name__ == "__main__":
    main()
