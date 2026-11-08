#!/usr/bin/env python3
"""
GreenVLM: DocVQA 20-Sample Energy-Accuracy Pareto Curve Benchmark.
Evaluates moondream:latest across 3 resolutions (224x224, 448x448, 896x896)
Computes ANLS (Average Normalized Levenshtein Similarity) & Energy in Joules.
Generates publication-quality Pareto curve visualization.
"""

import os
import sys
import time
import json
import base64
import urllib.request
from io import BytesIO
from typing import List, Dict, Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Import power tracker from previous module
from run_energy_experiment import BackgroundPowerTracker, can_read_rapl


# -------------------------------------------------------------
# 1. ANLS (Average Normalized Levenshtein Similarity) Metric
# -------------------------------------------------------------
def levenshtein_distance(s1: str, s2: str) -> int:
    """Standard Levenshtein Edit Distance."""
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev[j + 1] + 1
            dele = curr[j] + 1
            sub = prev[j] + (c1 != c2)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]

def compute_anls(prediction: str, ground_truths: List[str], threshold: float = 0.5) -> float:
    """
    Computes official DocVQA ANLS metric:
    score = 1 - (edit_dist / max_len) if edit_dist / max_len <= 0.5 else 0.0
    """
    if not prediction or not ground_truths:
        return 0.0
    pred = prediction.strip().lower()
    scores = []
    for gt in ground_truths:
        gt_clean = gt.strip().lower()
        dist = levenshtein_distance(pred, gt_clean)
        max_len = max(len(pred), len(gt_clean))
        if max_len == 0:
            scores.append(1.0)
            continue
        nl = dist / max_len
        scores.append(1.0 - nl if nl <= threshold else 0.0)
    return max(scores) if scores else 0.0


# -------------------------------------------------------------
# 2. Ollama Query Helper
# -------------------------------------------------------------
def query_ollama(model: str, prompt: str, image_b64: str, max_tokens: int = 25) -> Dict[str, Any]:
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64]
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": max_tokens
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        if "message" in res and "content" in res["message"]:
            res["response"] = res["message"]["content"]
        return res

def resize_to_b64(img_path: str, target_size: tuple) -> str:
    with Image.open(img_path) as img:
        img_res = img.resize(target_size, Image.Resampling.LANCZOS)
        buf = BytesIO()
        img_res.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# -------------------------------------------------------------
# 3. Main Benchmark Execution
# -------------------------------------------------------------
def run_docvqa_benchmark(
    samples_path: str = "docvqa_samples/samples.json",
    model_name: str = "moondream:latest",
    resolutions = [(224, 224), (448, 448), (896, 896)],
    num_samples: int = 20
):
    print("=" * 80)
    print("     GREEN-VLM: DocVQA 20-SAMPLE ENERGY-ACCURACY PARETO BENCHMARK")
    print("=" * 80)
    print(f"Model       : {model_name}")
    print(f"Resolutions : {[f'{r[0]}x{r[1]}' for r in resolutions]}")
    print(f"RAPL Mode   : {'Direct Hardware microjoules' if can_read_rapl() else 'Dynamic CPU Load Model'}")
    print("=" * 80)

    with open(samples_path, "r") as f:
        samples = json.load(f)[:num_samples]

    print(f"[*] Loaded {len(samples)} DocVQA samples.")

    # Storage for per-resolution summary
    res_aggregates = []
    full_log = []

    for res in resolutions:
        res_str = f"{res[0]}x{res[1]}"
        print(f"\n>>> Running Resolution: {res_str} ({len(samples)} samples) ...")

        anls_scores = []
        see_times = []
        speak_times = []
        tot_energies = []
        see_energies = []
        speak_energies = []
        in_tokens_list = []
        out_tokens_list = []

        for idx, s in enumerate(samples, start=1):
            q = s["question"]
            gts = s["answers"]
            img_path = s["image_path"]

            # DocVQA prompt: concise answer
            prompt = f"Question: {q}\nAnswer the question directly and concisely using only words from the document."

            img_b64 = resize_to_b64(img_path, res)

            tracker = BackgroundPowerTracker(interval_sec=0.02)
            tracker.start()

            try:
                resp = query_ollama(model=model_name, prompt=prompt, image_b64=img_b64, max_tokens=20)
            except Exception as e:
                tracker.stop()
                print(f"    [Error] Sample {idx} failed: {e}")
                continue

            p_stats = tracker.stop()

            # Timings
            t_see = resp.get("prompt_eval_duration", 0) / 1e9
            t_speak = resp.get("eval_duration", 0) / 1e9
            tot_j = p_stats["energy_joules"]

            t_active = max(t_see + t_speak, 0.001)
            rho_see = t_see / t_active
            see_j = tot_j * rho_see
            speak_j = tot_j * (1.0 - rho_see)

            ans = resp.get("response", "").strip()
            anls = compute_anls(ans, gts)

            in_toks = resp.get("prompt_eval_count", 0)
            out_toks = resp.get("eval_count", 0)

            anls_scores.append(anls)
            see_times.append(t_see)
            speak_times.append(t_speak)
            tot_energies.append(tot_j)
            see_energies.append(see_j)
            speak_energies.append(speak_j)
            in_tokens_list.append(in_toks)
            out_tokens_list.append(out_toks)

            full_log.append({
                "sample_id": s["id"],
                "resolution": res_str,
                "question": q,
                "ground_truth": gts,
                "prediction": ans,
                "anls": anls,
                "t_see": t_see,
                "t_speak": t_speak,
                "see_joules": see_j,
                "speak_joules": speak_j,
                "tot_joules": tot_j
            })

            if idx % 5 == 0 or idx == len(samples):
                print(f"    [{idx:02d}/{len(samples)}] ANLS: {np.mean(anls_scores)*100:4.1f}% | Avg See: {np.mean(see_times):.2f}s | Avg Speak: {np.mean(speak_times):.2f}s | Avg Energy: {np.mean(tot_energies):.1f}J")

        mean_anls = float(np.mean(anls_scores)) * 100.0
        mean_tot_j = float(np.mean(tot_energies))
        mean_see_j = float(np.mean(see_energies))
        mean_speak_j = float(np.mean(speak_energies))
        mean_see_t = float(np.mean(see_times))
        mean_speak_t = float(np.mean(speak_times))
        mean_in_tok = float(np.mean(in_tokens_list))
        mean_out_tok = float(np.mean(out_tokens_list))
        rho_pre = (mean_see_j / mean_tot_j * 100.0) if mean_tot_j > 0 else 0.0

        res_aggregates.append({
            "resolution": res_str,
            "res_dim": res[0],
            "anls": mean_anls,
            "total_joules": mean_tot_j,
            "see_joules": mean_see_j,
            "speak_joules": mean_speak_j,
            "see_time": mean_see_t,
            "speak_time": mean_speak_t,
            "in_tokens": mean_in_tok,
            "out_tokens": mean_out_tok,
            "rho_pre": rho_pre
        })

    # ---------------------------------------------------------
    # Summary Table
    # ---------------------------------------------------------
    print("\n" + "=" * 94)
    print("                 DocVQA (20 Samples) ENERGY-ACCURACY RESULTS TABLE")
    print("=" * 94)
    print(f"{'Resolution':<12} | {'ANLS (%)':<10} | {'See Time':<10} | {'Speak Time':<12} | {'See (J)':<10} | {'Speak (J)':<11} | {'Total (J)':<10} | {'See %':<8}")
    print("-" * 94)
    for agg in res_aggregates:
        print(f"{agg['resolution']:<12} | {agg['anls']:<10.2f} | {agg['see_time']:<10.2f} | {agg['speak_time']:<12.2f} | {agg['see_joules']:<10.2f} | {agg['speak_joules']:<11.2f} | {agg['total_joules']:<10.2f} | {agg['rho_pre']:<7.1f}%")
    print("=" * 94)

    # Save JSON results
    with open("docvqa_benchmark_results.json", "w") as f:
        json.dump({"aggregates": res_aggregates, "details": full_log}, f, indent=2)
    print("\n[✓] Saved detailed raw benchmark data to 'docvqa_benchmark_results.json'")

    # ---------------------------------------------------------
    # Plotting the Energy-Accuracy Pareto Curve
    # ---------------------------------------------------------
    plot_pareto_curve(res_aggregates, output_png="docvqa_energy_accuracy_pareto.png")
    return res_aggregates


def plot_pareto_curve(aggregates: List[Dict[str, Any]], output_png: str = "docvqa_energy_accuracy_pareto.png"):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Subplot 1: Energy-Accuracy Pareto Curve
    energies = [a["total_joules"] for a in aggregates]
    accuracies = [a["anls"] for a in aggregates]
    labels = [a["resolution"] for a in aggregates]

    ax1.plot(energies, accuracies, marker="o", markersize=9, linewidth=2.5, color="#1E88E5", label="Resolution Sweep")
    ax1.fill_between(energies, accuracies, color="#1E88E5", alpha=0.15)

    # Annotate points
    for x, y, lab in zip(energies, accuracies, labels):
        ax1.annotate(
            f"{lab}\n({y:.1f}% ANLS, {x:.1f}J)",
            xy=(x, y),
            xytext=(10, -15),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFFFFF", ec="#1E88E5", lw=1.2)
        )

    ax1.set_title("DocVQA: Energy-Accuracy Pareto Frontier", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Mean Inference Energy per Question (Joules)", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("Task Performance: ANLS Accuracy (%)", fontsize=11, fontweight="semibold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True)

    # Subplot 2: Stacked Energy Phase Breakdown (Seeing vs Speaking)
    x_indices = np.arange(len(labels))
    see_j = [a["see_joules"] for a in aggregates]
    speak_j = [a["speak_joules"] for a in aggregates]

    bar_width = 0.45
    p1 = ax2.bar(x_indices, see_j, width=bar_width, color="#00ACC1", label="Seeing Energy (Prefill / Vision)")
    p2 = ax2.bar(x_indices, speak_j, width=bar_width, bottom=see_j, color="#FB8C00", label="Speaking Energy (Decode / Text)")

    # Label prefill percentages on bars
    for idx, (sj, spj, agg) in enumerate(zip(see_j, speak_j, aggregates)):
        tot = sj + spj
        see_pct = (sj / tot) * 100
        ax2.text(idx, tot + 1.0, f"Total: {tot:.1f}J\n(Seeing: {see_pct:.1f}%)", ha="center", fontsize=9, fontweight="bold")

    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax2.set_title("Energy Breakdown: Seeing (Prefill) vs. Speaking (Decode)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Input Resolution", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Energy per Inference (Joules)", fontsize=11, fontweight="semibold")
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"[✓] Successfully generated Pareto plot: '{output_png}'")


if __name__ == "__main__":
    run_docvqa_benchmark()
    os._exit(0)
