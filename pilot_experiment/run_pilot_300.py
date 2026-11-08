#!/usr/bin/env python3
"""
GreenVLM: 100-Sample Multi-Task Energy-Accuracy Benchmark.
Evaluates 4 Task Archetypes across 3 Resolutions (224x224, 448x448, 896x896):
1. DocVQA (25 samples) - Fine-grained Document OCR (Vision-Dominant)
2. ChartQA (25 samples) - Complex Visual Data Reasoning (Crossover Parity)
3. POPE (25 samples) - Object Probing / Hallucination (Zero-Drift 1-Token Control)
4. COCO (25 samples) - Open Captioning (Decode-Dominant Baseline)

Total: 100 samples x 3 resolutions = 300 inferences.
Generates the Signature Phase-Transition Plot & Multi-Task Pareto Curves.
"""

import os
import re
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

from run_energy_experiment import BackgroundPowerTracker, can_read_rapl
from benchmark_docvqa_pareto import levenshtein_distance, compute_anls


# -------------------------------------------------------------
# 1. Task-Specific Accuracy Evaluators
# -------------------------------------------------------------
def eval_chartqa(prediction: str, ground_truths: List[str]) -> float:
    """Evaluates ChartQA relaxed accuracy (numeric 5% tolerance or string match)."""
    if not prediction or not ground_truths:
        return 0.0
    pred = prediction.strip().lower()
    for gt in ground_truths:
        gt_clean = str(gt).strip().lower()
        if pred == gt_clean:
            return 1.0
        # Check numeric tolerance
        try:
            p_nums = re.findall(r"[-+]?\d*\.\d+|\d+", pred)
            g_nums = re.findall(r"[-+]?\d*\.\d+|\d+", gt_clean)
            if p_nums and g_nums:
                p_val = float(p_nums[0])
                g_val = float(g_nums[0])
                if g_val != 0:
                    rel_diff = abs(p_val - g_val) / abs(g_val)
                    if rel_diff <= 0.05:
                        return 1.0
                elif p_val == g_val:
                    return 1.0
        except Exception:
            pass
        # Substring match for short textual labels
        if len(gt_clean) > 2 and (gt_clean in pred or pred in gt_clean):
            return 1.0
    return 0.0

def eval_pope(prediction: str, ground_truth: str) -> float:
    """Evaluates POPE binary match ('yes' / 'no')."""
    pred = prediction.strip().lower()
    gt = ground_truth.strip().lower()
    
    # Check for direct word presence
    has_yes = bool(re.search(r"\byes\b", pred))
    has_no = bool(re.search(r"\bno\b", pred))
    
    if has_yes and not has_no:
        extracted = "yes"
    elif has_no and not has_yes:
        extracted = "no"
    else:
        extracted = pred[:3]

    return 1.0 if extracted == gt else 0.0


# -------------------------------------------------------------
# 2. Ollama Query & Image Helper
# -------------------------------------------------------------
def query_ollama_chat(model: str, prompt: str, image_b64: str, max_tokens: int = 25) -> Dict[str, Any]:
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
# 3. Master Multi-Task Runner
# -------------------------------------------------------------
def run_multitask_benchmark(
    meta_path: str = "multitask_samples/multitask_samples_metadata.json",
    model_name: str = "moondream:latest",
    resolutions = [(224, 224), (448, 448), (896, 896)]
):
    print("=" * 86)
    print("    GREEN-VLM: 100-SAMPLE MULTI-TASK ENERGY & ACCURACY BENCHMARK (OPTION B)")
    print("=" * 86)
    print(f"Model       : {model_name}")
    print(f"Tasks       : DocVQA (25), ChartQA (25), POPE (25), COCO (25) = 100 Samples")
    print(f"Resolutions : {[f'{r[0]}x{r[1]}' for r in resolutions]} -> Total 300 Inferences")
    print(f"RAPL Access : {'Hardware microjoules register' if can_read_rapl() else 'Dynamic CPU Power Model'}")
    print("=" * 86)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    task_order = ["DocVQA", "ChartQA", "POPE", "COCO"]
    task_max_tokens = {
        "DocVQA": 20,
        "ChartQA": 15,
        "POPE": 5,
        "COCO": 45
    }

    full_results = []
    task_summary = {t: {} for t in task_order}

    overall_start_time = time.time()
    total_inferences = len(task_order) * 25 * len(resolutions)
    inf_count = 0

    for task_name in task_order:
        samples = meta.get(task_name, [])
        print(f"\n" + "#" * 70)
        print(f"  EXECUTING TASK: {task_name.upper()} ({len(samples)} SAMPLES)")
        print("#" * 70)

        for res in resolutions:
            res_str = f"{res[0]}x{res[1]}"
            print(f"\n  >>> Resolution: {res_str} | Task: {task_name} ...")

            accuracies = []
            see_times = []
            speak_times = []
            tot_energies = []
            see_energies = []
            speak_energies = []
            in_toks_list = []
            out_toks_list = []

            for idx, s in enumerate(samples, start=1):
                inf_count += 1
                q = s["question"]
                gts = s["ground_truth"]
                img_path = s["image_path"]

                # Task-specific prompt formatting
                if task_name == "DocVQA":
                    prompt = f"Question: {q}\nAnswer the question directly and concisely using words from the document."
                elif task_name == "ChartQA":
                    prompt = f"Question: {q}\nAnswer concisely with the exact number or label shown in the chart."
                elif task_name == "POPE":
                    prompt = f"Question: {q}\nAnswer with exactly one word: 'yes' or 'no'."
                else:  # COCO
                    prompt = "Describe this image concisely in 2 sentences."

                img_b64 = resize_to_b64(img_path, res)

                tracker = BackgroundPowerTracker(interval_sec=0.02)
                tracker.start()

                try:
                    resp = query_ollama_chat(
                        model=model_name,
                        prompt=prompt,
                        image_b64=img_b64,
                        max_tokens=task_max_tokens[task_name]
                    )
                except Exception as e:
                    tracker.stop()
                    print(f"    [Error] Sample {idx} failed: {e}")
                    continue

                p_stats = tracker.stop()

                t_see = resp.get("prompt_eval_duration", 0) / 1e9
                t_speak = resp.get("eval_duration", 0) / 1e9
                tot_j = p_stats["energy_joules"]

                t_active = max(t_see + t_speak, 0.001)
                rho_see = t_see / t_active
                see_j = tot_j * rho_see
                speak_j = tot_j * (1.0 - rho_see)

                ans = resp.get("response", "").strip()
                in_toks = resp.get("prompt_eval_count", 0)
                out_toks = resp.get("eval_count", 0)

                # Compute Accuracy
                if task_name == "DocVQA":
                    acc = compute_anls(ans, gts)
                elif task_name == "ChartQA":
                    acc = eval_chartqa(ans, gts)
                elif task_name == "POPE":
                    acc = eval_pope(ans, gts[0] if gts else "no")
                else:  # COCO
                    acc = float(out_toks)  # For COCO, track generated token length

                accuracies.append(acc)
                see_times.append(t_see)
                speak_times.append(t_speak)
                tot_energies.append(tot_j)
                see_energies.append(see_j)
                speak_energies.append(speak_j)
                in_toks_list.append(in_toks)
                out_toks_list.append(out_toks)

                full_results.append({
                    "task": task_name,
                    "sample_id": s["id"],
                    "resolution": res_str,
                    "question": q,
                    "ground_truth": gts,
                    "prediction": ans,
                    "score": acc,
                    "t_see": t_see,
                    "t_speak": t_speak,
                    "see_j": see_j,
                    "speak_j": speak_j,
                    "tot_j": tot_j,
                    "in_toks": in_toks,
                    "out_toks": out_toks
                })

                if idx % 5 == 0 or idx == len(samples):
                    acc_label = "Avg Toks" if task_name == "COCO" else "Score"
                    print(f"    [{idx:02d}/{len(samples)}] Progress: {inf_count}/{total_inferences} | {acc_label}: {np.mean(accuracies)*(1 if task_name=='COCO' else 100):4.1f}{'' if task_name=='COCO' else '%'} | See: {np.mean(see_times):.2f}s | Speak: {np.mean(speak_times):.2f}s | Energy: {np.mean(tot_energies):.1f}J")

            # Store aggregated summary for this (task, res)
            mean_acc = float(np.mean(accuracies)) * (1.0 if task_name == "COCO" else 100.0)
            mean_tot_j = float(np.mean(tot_energies))
            mean_see_j = float(np.mean(see_energies))
            mean_speak_j = float(np.mean(speak_energies))
            mean_see_t = float(np.mean(see_times))
            mean_speak_t = float(np.mean(speak_times))
            mean_out_tok = float(np.mean(out_toks_list))
            rho_pre = (mean_see_j / mean_tot_j * 100.0) if mean_tot_j > 0 else 0.0

            task_summary[task_name][res_str] = {
                "accuracy": mean_acc,
                "see_j": mean_see_j,
                "speak_j": mean_speak_j,
                "tot_j": mean_tot_j,
                "see_t": mean_see_t,
                "speak_t": mean_speak_t,
                "out_toks": mean_out_tok,
                "rho_pre": rho_pre
            }

    elapsed_min = (time.time() - overall_start_time) / 60.0
    print(f"\n[✓] All 300 inferences completed in {elapsed_min:.1f} minutes!")

    # ---------------------------------------------------------
    # Consolidated Master Summary Table
    # ---------------------------------------------------------
    print("\n" + "=" * 98)
    print("                     100-SAMPLE MULTI-TASK SUMMARY RESULTS TABLE")
    print("=" * 98)
    print(f"{'Task':<10} | {'Resolution':<10} | {'Metric / Acc':<14} | {'Out Toks':<9} | {'See Time':<9} | {'Speak Time':<11} | {'See (J)':<9} | {'Speak (J)':<10} | {'See Share (%)':<12}")
    print("-" * 98)
    for t in task_order:
        for res_str in ["224x224", "448x448", "896x896"]:
            s = task_summary[t][res_str]
            metric_str = f"{s['accuracy']:.1f} toks" if t == "COCO" else f"{s['accuracy']:.2f}%"
            print(f"{t:<10} | {res_str:<10} | {metric_str:<14} | {s['out_toks']:<9.1f} | {s['see_t']:<9.2f} | {s['speak_t']:<11.2f} | {s['see_j']:<9.1f} | {s['speak_j']:<10.1f} | {s['rho_pre']:<11.1f}%")
        print("-" * 98)

    # Save complete raw data
    with open("multitask_benchmark_results.json", "w") as f:
        json.dump({"summary": task_summary, "details": full_results}, f, indent=2)
    print("\n[✓] Saved complete results to 'multitask_benchmark_results.json'")

    # ---------------------------------------------------------
    # Publication-Quality Multi-Panel Plot
    # ---------------------------------------------------------
    plot_multitask_figures(task_summary, output_png="multitask_phase_transition_and_pareto.png")
    return task_summary


def plot_multitask_figures(summary: Dict[str, Any], output_png: str = "multitask_phase_transition_and_pareto.png"):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # ---------------------------------------------------------
    # Panel 1: The Seeing vs Speaking Phase Transition
    # ---------------------------------------------------------
    tasks = ["COCO\n(Captioning)", "ChartQA\n(Reasoning)", "POPE\n(Probing)", "DocVQA\n(Doc OCR)"]
    task_keys = ["COCO", "ChartQA", "POPE", "DocVQA"]
    
    # We display at 448x448 (standard VLM default resolution)
    res_key = "448x448"
    see_shares = [summary[k][res_key]["rho_pre"] for k in task_keys]
    speak_shares = [100.0 - s for s in see_shares]

    y_pos = np.arange(len(tasks))
    bar_height = 0.52

    p1 = ax1.barh(y_pos, see_shares, height=bar_height, color="#00ACC1", label="Seeing Share (Prefill / Vision)")
    p2 = ax1.barh(y_pos, speak_shares, height=bar_height, left=see_shares, color="#FB8C00", label="Speaking Share (Decode / Text)")

    # 50% Threshold Crossover Line
    ax1.axvline(50, color="#D32F2F", linestyle="--", linewidth=2.0, label="50% Crossover Boundary")

    # Percentage Labels inside bars
    for idx, (s_share, sp_share) in enumerate(zip(see_shares, speak_shares)):
        if s_share > 15:
            ax1.text(s_share / 2, idx, f"{s_share:.1f}%", va="center", ha="center", color="white", fontweight="bold", fontsize=10.5)
        if sp_share > 15:
            ax1.text(s_share + sp_share / 2, idx, f"{sp_share:.1f}%", va="center", ha="center", color="white", fontweight="bold", fontsize=10.5)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(tasks, fontsize=11, fontweight="bold")
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Energy Share (%) at 448x448 Resolution", fontsize=11, fontweight="semibold")
    ax1.set_title("The 'Seeing vs. Speaking' Phase Transition Across Tasks", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # ---------------------------------------------------------
    # Panel 2: Multi-Task Energy-Accuracy Pareto Curves
    # ---------------------------------------------------------
    colors = {"DocVQA": "#1E88E5", "ChartQA": "#43A047", "POPE": "#8E24AA"}
    markers = {"DocVQA": "o", "ChartQA": "s", "POPE": "^"}

    for t_key in ["DocVQA", "ChartQA", "POPE"]:
        energies = [summary[t_key][r]["tot_j"] for r in ["224x224", "448x448", "896x896"]]
        accs = [summary[t_key][r]["accuracy"] for r in ["224x224", "448x448", "896x896"]]
        ax2.plot(energies, accs, marker=markers[t_key], markersize=8, linewidth=2.2, color=colors[t_key], label=f"{t_key}")
        
        # Annotate resolution points
        for e, a, res_lbl in zip(energies, accs, ["224", "448", "896"]):
            ax2.annotate(f"{res_lbl}px", (e, a), textcoords="offset points", xytext=(8, -4), fontsize=8.5, fontweight="bold", color=colors[t_key])

    ax2.set_xlabel("Mean Inference Energy per Query (Joules)", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Task Accuracy (%)", fontsize=11, fontweight="semibold")
    ax2.set_title("Energy-Accuracy Pareto Frontiers across Resolutions", fontsize=13, fontweight="bold", pad=12)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved publication plot: '{output_png}'")


if __name__ == "__main__":
    run_multitask_benchmark()
    os._exit(0)
