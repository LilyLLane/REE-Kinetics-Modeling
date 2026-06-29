"""
Kd and Extraction Efficiency Calculator for Batch Extraction ICP Data
======================================================================
Experiment: Lanthanide LLE with TODGA / 3M HCl, fixed Ln 0.5 mM
Phase ratio: S/F = 1 (0.75 mL organic : 0.75 mL aqueous)
ICP sample prep: 0.5 mL raffinate diluted into 4 mL total (DF = 8)

Equations:
    C_f  = C_measured * DF                  (undo ICP dilution)
    C_i  = 0.5 mM * MW_element (mg/L = ppm) (nominal initial concentration)
    E    = (C_i - C_f) / C_i                (extraction efficiency, 0–1)
    K_d  = C_f / (C_i - C_f)               (= (1-E)/E, valid for S/F = 1)

Uncertainty propagation (replicate std → propagated through E and Kd):
    sigma_E   = sigma_Cf / C_i              (C_i assumed exact)
    sigma_Kd  = sigma_E / E^2              (from dKd/dE = -1/E^2)
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
CSV_PATH     = Path("/mnt/user-data/uploads/20260611-10-60min-_fixed_Ln_0_5_mM_-_3M_HCl_CSV.csv")
OUTPUT_DIR   = Path("/mnt/user-data/outputs")
DILUTION_FACTOR = 8        # 0.5 mL sample into 4 mL total
INITIAL_CONC_MM = 0.5      # mM, nominal initial aqueous Ln concentration
SF_RATIO     = 1.0         # organic : aqueous phase ratio

# Lanthanide molar masses (g/mol) → 0.5 mM * MW = ppm
MOLAR_MASS = {
    "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24,
    "Sm": 150.36, "Eu": 151.96, "Gd": 157.25, "Tb": 158.93,
    "Dy": 162.50, "Ho": 164.93, "Er": 167.26, "Tm": 168.93,
    "Yb": 173.05, "Lu": 174.97,
}

# Lanthanide series order for plotting
LN_ORDER = ["La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu"]

# ── Load and parse data ────────────────────────────────────────────────────────
def load_icp(path):
    df = pd.read_csv(path, skiprows=6, header=0)
    df = df[df["Type"] == "Sample"].copy()
    df = df[df["Label"] != "blank"].copy()
    df["Concentration"] = pd.to_numeric(df["Concentration"], errors="coerce")
    df["Concentration SD"] = pd.to_numeric(df["Concentration SD"], errors="coerce")
    return df


def parse_label(label):
    m = re.match(r"[Bb][Ee]-(\d+\.?\d*)-(\d+)min-(\d+)", label)
    if m:
        return float(m.group(1)), int(m.group(2)), int(m.group(3))
    return np.nan, np.nan, np.nan


def compute_kd(df):
    df[["TODGA_M", "time_min", "replicate"]] = df["Label"].apply(
        lambda l: pd.Series(parse_label(l))
    )

    # Undilute: actual raffinate concentration
    df["C_f_ppm"] = df["Concentration"] * DILUTION_FACTOR
    df["C_f_sd_ppm"] = df["Concentration SD"] * DILUTION_FACTOR

    # Nominal initial concentration per element
    df["C_i_ppm"] = df["Element Label"].map(
        {el: INITIAL_CONC_MM * mw for el, mw in MOLAR_MASS.items()}
    )

    # Extraction efficiency and Kd per replicate
    df["E"]  = (df["C_i_ppm"] - df["C_f_ppm"]) / df["C_i_ppm"]
    df["Kd"] = df["C_f_ppm"] / (df["C_i_ppm"] - df["C_f_ppm"])

    # Propagated uncertainty (from C_f SD only; C_i treated as exact)
    df["sigma_E"]  = df["C_f_sd_ppm"] / df["C_i_ppm"]
    df["sigma_Kd"] = df["sigma_E"] / (df["E"] ** 2)

    return df


def average_replicates(df):
    grouped = df.groupby(["Element Label", "TODGA_M", "time_min"])
    out = grouped.agg(
        C_f_mean   = ("C_f_ppm",  "mean"),
        C_f_std    = ("C_f_ppm",  "std"),
        C_i_ppm    = ("C_i_ppm",  "first"),
        E_mean     = ("E",        "mean"),
        E_std      = ("E",        "std"),
        Kd_mean    = ("Kd",       "mean"),
        Kd_std     = ("Kd",       "std"),
        n          = ("Kd",       "count"),
    ).reset_index()

    # Standard error across replicates
    out["E_se"]  = out["E_std"]  / np.sqrt(out["n"])
    out["Kd_se"] = out["Kd_std"] / np.sqrt(out["n"])

    # Enforce lanthanide order
    out["Element Label"] = pd.Categorical(
        out["Element Label"], categories=LN_ORDER, ordered=True
    )
    out = out.sort_values(["Element Label", "TODGA_M", "time_min"])
    return out


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_kd_vs_element(results, todga_conc, time_min, ax, color, label):
    sub = results[
        (np.isclose(results["TODGA_M"], todga_conc)) &
        (results["time_min"] == time_min)
    ].sort_values("Element Label")
    ax.errorbar(
        sub["Element Label"], sub["Kd_mean"], yerr=sub["Kd_se"],
        fmt="o-", color=color, label=label, capsize=3, linewidth=1.5, markersize=5
    )


def make_kd_vs_ln_plot(results, output_dir):
    """K_d across lanthanide series for each TODGA concentration at 60 min."""
    todga_vals = sorted(results["TODGA_M"].unique())
    colors = cm.viridis(np.linspace(0.15, 0.85, len(todga_vals)))

    fig, ax = plt.subplots(figsize=(10, 5))
    for todga, color in zip(todga_vals, colors):
        sub = results[
            (np.isclose(results["TODGA_M"], todga)) &
            (results["time_min"] == 60)
        ].sort_values("Element Label")
        if sub.empty:
            continue
        ax.errorbar(
            sub["Element Label"], sub["Kd_mean"], yerr=sub["Kd_se"],
            fmt="o-", color=color, label=f"{todga} M TODGA",
            capsize=3, linewidth=1.5, markersize=5
        )

    ax.set_xlabel("Lanthanide Element")
    ax.set_ylabel("K$_d$")
    ax.set_title("Distribution Coefficient across Lanthanide Series\n(60 min contact time, 3M HCl, S/F = 1)")
    ax.legend(title="[TODGA]", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "kd_vs_lanthanide_60min.png", dpi=150)
    plt.close(fig)
    print("Saved: kd_vs_lanthanide_60min.png")


def make_kd_vs_todga_plot(results, output_dir):
    """K_d vs TODGA concentration for each element at 60 min."""
    elements = [e for e in LN_ORDER if e in results["Element Label"].values]
    colors = cm.tab20(np.linspace(0, 1, len(elements)))

    fig, ax = plt.subplots(figsize=(9, 5))
    for el, color in zip(elements, colors):
        sub = results[
            (results["Element Label"] == el) &
            (results["time_min"] == 60)
        ].sort_values("TODGA_M")
        if sub.empty:
            continue
        ax.errorbar(
            sub["TODGA_M"], sub["Kd_mean"], yerr=sub["Kd_se"],
            fmt="o-", color=color, label=el,
            capsize=3, linewidth=1.5, markersize=5
        )

    ax.set_xlabel("[TODGA] (M)")
    ax.set_ylabel("K$_d$")
    ax.set_title("K$_d$ vs [TODGA] by Element\n(60 min contact time, 3M HCl, S/F = 1)")
    ax.legend(title="Element", bbox_to_anchor=(1.01, 1), loc="upper left", ncol=1, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "kd_vs_todga_60min.png", dpi=150)
    plt.close(fig)
    print("Saved: kd_vs_todga_60min.png")


def make_E_heatmap(results, output_dir):
    """Heatmap of mean extraction efficiency at 60 min."""
    sub = results[results["time_min"] == 60].copy()
    pivot = sub.pivot_table(
        index="Element Label", columns="TODGA_M", values="E_mean"
    )
    # Enforce lanthanide row order
    pivot = pivot.reindex([e for e in LN_ORDER if e in pivot.index])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.3f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("[TODGA] (M)")
    ax.set_ylabel("Element")
    ax.set_title("Extraction Efficiency E\n(60 min contact time, 3M HCl, S/F = 1)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black")
    plt.colorbar(im, ax=ax, label="E")
    fig.tight_layout()
    fig.savefig(output_dir / "E_heatmap_60min.png", dpi=150)
    plt.close(fig)
    print("Saved: E_heatmap_60min.png")


def make_kd_vs_time_plot(results, output_dir):
    """K_d vs contact time for each element at a mid-range TODGA concentration."""
    # Use 0.05 M as the representative mid-range (has all 4 time points)
    todga_ref = 0.05
    elements = [e for e in LN_ORDER if e in results["Element Label"].values]
    colors = cm.tab20(np.linspace(0, 1, len(elements)))

    fig, ax = plt.subplots(figsize=(9, 5))
    for el, color in zip(elements, colors):
        sub = results[
            (results["Element Label"] == el) &
            (np.isclose(results["TODGA_M"], todga_ref))
        ].sort_values("time_min")
        if sub.empty:
            continue
        ax.errorbar(
            sub["time_min"], sub["Kd_mean"], yerr=sub["Kd_se"],
            fmt="o-", color=color, label=el,
            capsize=3, linewidth=1.5, markersize=5
        )

    ax.set_xlabel("Contact Time (min)")
    ax.set_ylabel("K$_d$")
    ax.set_title(f"K$_d$ vs Contact Time by Element\n([TODGA] = {todga_ref} M, 3M HCl, S/F = 1)")
    ax.legend(title="Element", bbox_to_anchor=(1.01, 1), loc="upper left", ncol=1, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "kd_vs_time_0.05M_TODGA.png", dpi=150)
    plt.close(fig)
    print("Saved: kd_vs_time_0.05M_TODGA.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ICP data...")
    raw = load_icp(CSV_PATH)

    print("Computing K_d and E per replicate...")
    per_rep = compute_kd(raw)

    print("Averaging replicates...")
    results = average_replicates(per_rep)

    # Save full results table
    csv_out = OUTPUT_DIR / "kd_results.csv"
    results.to_csv(csv_out, index=False, float_format="%.4f")
    print(f"Saved: kd_results.csv ({len(results)} rows)")

    # Save per-replicate table too
    rep_out = OUTPUT_DIR / "kd_per_replicate.csv"
    per_rep[["Label","Element Label","TODGA_M","time_min","replicate",
             "C_i_ppm","C_f_ppm","C_f_sd_ppm","E","sigma_E","Kd","sigma_Kd"]].to_csv(
        rep_out, index=False, float_format="%.4f"
    )
    print(f"Saved: kd_per_replicate.csv ({len(per_rep)} rows)")

    # Plots
    print("\nGenerating plots...")
    make_kd_vs_ln_plot(results, OUTPUT_DIR)
    make_kd_vs_todga_plot(results, OUTPUT_DIR)
    make_E_heatmap(results, OUTPUT_DIR)
    make_kd_vs_time_plot(results, OUTPUT_DIR)

    # Print a summary table for La, Nd, Eu, Yb at 60 min
    print("\n── Sample results (60 min, selected elements) ──────────────────────")
    summary = results[
        (results["time_min"] == 60) &
        (results["Element Label"].isin(["La", "Nd", "Eu", "Yb"]))
    ][["Element Label","TODGA_M","C_i_ppm","C_f_mean","E_mean","E_std","Kd_mean","Kd_std"]]
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
