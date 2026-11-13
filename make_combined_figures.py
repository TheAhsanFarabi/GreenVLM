#!/usr/bin/env python3
"""
GreenVLM: cross-model combined results.

Reads every <results-dir>/<model>/benchmark_results.json and emits:
  - combined_summary.csv                 (flat table of every task x resolution x model)
  - combined_phase_transition.png        (seeing-energy share by model across tasks)
  - combined_pareto.png                  (energy vs accuracy Pareto, short-answer tasks)
  - combined_rho_vs_resolution.png       (phase-transition curves: seeing share vs resolution)

Usage:
    python make_combined_figures.py --results-dir results/full_v2
"""
import os, json, glob, argparse, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DISPLAY = {
    "qwen3-vl_8b": "Qwen3-VL-8B",
    "gemma4_12b": "Gemma4-12B",
    "minicpm-v_8b": "MiniCPM-V-8B",
    "ministral-3_8b": "Ministral-3-8B",
    "llava_7b": "LLaVA-7B",
    "OpenGVLab_InternVL3-8B-hf": "InternVL3-8B",
}
ORDER = ["InternVL3-8B", "Gemma4-12B", "MiniCPM-V-8B", "Ministral-3-8B", "LLaVA-7B", "Qwen3-VL-8B"]
COLORS = {
    "InternVL3-8B": "#1E88E5", "Gemma4-12B": "#43A047", "MiniCPM-V-8B": "#8E24AA",
    "Ministral-3-8B": "#FB8C00", "LLaVA-7B": "#6D4C41", "Qwen3-VL-8B": "#E53935",
}
TASKS = ["DocVQA", "ChartQA", "POPE", "COCO"]
SHORT_TASKS = ["DocVQA", "ChartQA", "POPE"]


def load(results_dir):
    data = {}
    for p in glob.glob(os.path.join(results_dir, "*", "benchmark_results.json")):
        key = os.path.basename(os.path.dirname(p))
        name = DISPLAY.get(key, key)
        data[name] = json.load(open(p))["summary"]
    return {n: data[n] for n in ORDER if n in data}


def write_csv(data, out):
    rows = []
    for model, s in data.items():
        for task in TASKS:
            for res, v in s.get(task, {}).items():
                rows.append({
                    "model": model, "task": task, "resolution": res,
                    "metric": v.get("metric", ""), "value": round(v["accuracy"], 3),
                    "seeing_pct": round(v["rho_pre"], 2), "see_j": round(v["see_j"], 2),
                    "speak_j": round(v["speak_j"], 2), "tot_j": round(v["tot_j"], 2),
                    "out_tokens": round(v["out_toks"], 1),
                })
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[saved] {out}  ({len(rows)} rows)")


def fig_phase_transition(data, out, res="448x448"):
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), dpi=200)
    models = list(data.keys())
    y = np.arange(len(models))
    for ax, task in zip(axes, TASKS):
        see = [data[m].get(task, {}).get(res, {}).get("rho_pre", np.nan) for m in models]
        ax.barh(y, see, color=[COLORS[m] for m in models], height=0.6)
        ax.axvline(50, color="#D32F2F", ls="--", lw=1.8)
        ax.set_yticks(y); ax.set_yticklabels(models if task == "DocVQA" else [])
        ax.set_xlim(0, 100); ax.invert_yaxis()
        ax.set_xlabel("Seeing-energy share (%)")
        ax.set_title(f"{task}  @ {res}", fontweight="bold")
        for i, s in enumerate(see):
            if not np.isnan(s):
                ax.text(min(s + 1.5, 92), i, f"{s:.0f}%", va="center", fontsize=9)
    fig.suptitle("The 'Seeing vs. Speaking' Phase Transition across Models (red line = 50% crossover)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"[saved] {out}")


def fig_pareto(data, out):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=200)
    for ax, task in zip(axes, SHORT_TASKS):
        for m, s in data.items():
            d = s.get(task, {})
            if not d:
                continue
            reses = sorted(d.keys(), key=lambda r: int(r.split("x")[0]))
            e = [d[r]["tot_j"] for r in reses]
            a = [d[r]["accuracy"] for r in reses]
            ax.plot(e, a, "-o", color=COLORS[m], lw=2, ms=7, label=m)
            for r, x, yy in zip(reses, e, a):
                ax.annotate(r.split("x")[0], (x, yy), textcoords="offset points",
                            xytext=(5, 4), fontsize=7.5)
        ax.set_xlabel("Mean energy per query (J)"); ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{task} — Energy vs Accuracy", fontweight="bold")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Energy–Accuracy Pareto Frontiers (labels = input resolution in px)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"[saved] {out}")


def fig_rho_vs_res(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)
    for ax, task in zip(axes, ["DocVQA", "ChartQA"]):
        for m, s in data.items():
            d = s.get(task, {})
            if not d:
                continue
            reses = sorted(d.keys(), key=lambda r: int(r.split("x")[0]))
            x = [int(r.split("x")[0]) for r in reses]
            rho = [d[r]["rho_pre"] for r in reses]
            ax.plot(x, rho, "-o", color=COLORS[m], lw=2, ms=7, label=m)
        ax.axhline(50, color="#D32F2F", ls="--", lw=1.8)
        ax.set_xticks([224, 448, 896]); ax.set_ylim(0, 100)
        ax.set_xlabel("Input resolution (px)"); ax.set_ylabel("Seeing-energy share (%)")
        ax.set_title(f"{task}: seeing share vs resolution", fontweight="bold")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Resolution-driven Phase Transition (crossing the 50% line = seeing becomes dominant)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94]); plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"[saved] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/full_v2")
    args = ap.parse_args()
    data = load(args.results_dir)
    if not data:
        print(f"[!] No benchmark_results.json found under {args.results_dir}"); return
    print("Models:", ", ".join(data.keys()))
    d = args.results_dir
    write_csv(data, os.path.join(d, "combined_summary.csv"))
    fig_phase_transition(data, os.path.join(d, "combined_phase_transition.png"))
    fig_pareto(data, os.path.join(d, "combined_pareto.png"))
    fig_rho_vs_res(data, os.path.join(d, "combined_rho_vs_resolution.png"))


if __name__ == "__main__":
    main()
