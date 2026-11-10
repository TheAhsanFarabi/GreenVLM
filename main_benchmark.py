#!/usr/bin/env python3
"""
GreenVLM: The Illusion of "Free" Vision in Real-World Multimodal Tasks.
Master Benchmarking Engine for Multimodal Inference Energy & Accuracy.

Supports:
- Any VLM via Ollama or HuggingFace (Moondream, Qwen2.5-VL, InternVL, Gemma-3, LLaVA)
- Multi-Task Evaluation (DocVQA, ChartQA, POPE, COCO)
- Resolution Sweeps (e.g. 224x224, 448x448, 896x896, 1344x1344)
- Automated Hardware Energy Profiling (NVIDIA GPU NVML, Intel RAPL, CPU fallback)
- Publication-quality Phase Transition & Pareto Frontier Plots

Usage:
    # Run with default model (Moondream) on cached pilot data
    python main_benchmark.py

    # Run with Qwen2.5-VL-7B on GPU
    python main_benchmark.py --model qwen2.5-vl:7b --tasks DocVQA ChartQA

    # Custom resolutions and output
    python main_benchmark.py --model qwen2.5-vl:7b --resolutions 448 896 1344 --output-dir results/
"""

import os
import re
import sys
import json
import base64
import argparse
import urllib.request
from io import BytesIO
from typing import List, Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from energy_profiler import HardwareEnergyTracker, can_read_rapl


# -------------------------------------------------------------
# Accuracy Evaluators
# -------------------------------------------------------------
def levenshtein_dist(s1: str, s2: str) -> int:
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if len(s1) < len(s2):
        return levenshtein_dist(s2, s1)
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

def compute_anls(pred: str, ground_truths: List[str], threshold: float = 0.5) -> float:
    if not pred or not ground_truths:
        return 0.0
    p = pred.strip().lower()
    scores = []
    for gt in ground_truths:
        g = str(gt).strip().lower()
        d = levenshtein_dist(p, g)
        m = max(len(p), len(g))
        if m == 0:
            scores.append(1.0)
            continue
        nl = d / m
        scores.append(1.0 - nl if nl <= threshold else 0.0)
    return max(scores) if scores else 0.0

def eval_chartqa(pred: str, ground_truths: List[str]) -> float:
    if not pred or not ground_truths:
        return 0.0
    p = pred.strip().lower()
    for gt in ground_truths:
        g = str(gt).strip().lower()
        if p == g or g in p or p in g:
            return 1.0
        try:
            pn = re.findall(r"[-+]?\d*\.\d+|\d+", p)
            gn = re.findall(r"[-+]?\d*\.\d+|\d+", g)
            if pn and gn:
                pv, gv = float(pn[0]), float(gn[0])
                if gv != 0 and abs(pv - gv) / abs(gv) <= 0.05:
                    return 1.0
                elif pv == gv:
                    return 1.0
        except Exception:
            pass
    return 0.0

def eval_pope(pred: str, ground_truth: str) -> float:
    p = pred.strip().lower()
    g = ground_truth.strip().lower()
    has_yes = bool(re.search(r"\byes\b", p))
    has_no = bool(re.search(r"\bno\b", p))
    if has_yes and not has_no:
        ext = "yes"
    elif has_no and not has_yes:
        ext = "no"
    else:
        ext = p[:3]
    return 1.0 if ext == g else 0.0


# -------------------------------------------------------------
# Ollama Client & Image Resizer
# -------------------------------------------------------------
def query_model(model: str, prompt: str, image_b64: str, max_tokens: int = 30) -> Dict[str, Any]:
    url = "http://localhost:11434/api/chat"
    base = model.split(":")[0].lower()

    # Reasoning VLMs (e.g. qwen3-vl) always emit chain-of-thought and only place the
    # final answer in message.content *after* thinking finishes, so they need a larger
    # decode budget. Their reasoning tokens are genuine decode work ("speaking" energy).
    reasoning = base.startswith("qwen3")
    num_predict = max(max_tokens + 256, 320) if reasoning else max_tokens

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": num_predict}
    }
    # gemma thinking models hide the answer unless thinking is explicitly disabled.
    # (llama3.2-vision is not a thinking model and errors if 'think' is sent.)
    if base.startswith("gemma"):
        payload["think"] = False

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        if "message" in res and "content" in res["message"]:
            res["response"] = res["message"]["content"]
        return res

def resize_image_b64(path: str, size: tuple) -> str:
    with Image.open(path) as img:
        img_res = img.resize(size, Image.Resampling.LANCZOS)
        buf = BytesIO()
        img_res.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# -------------------------------------------------------------
# Plotting Engine
# -------------------------------------------------------------
def generate_plots(summary: Dict[str, Any], output_path: str):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # 1. Phase Transition Plot
    task_keys = [k for k in ["COCO", "ChartQA", "POPE", "DocVQA"] if k in summary]
    task_labels = [f"{k}" for k in task_keys]
    res_default = "448x448" if "448x448" in list(summary[task_keys[0]].keys()) else list(summary[task_keys[0]].keys())[0]

    see_pcts = [summary[k][res_default]["rho_pre"] for k in task_keys]
    speak_pcts = [100.0 - s for s in see_pcts]
    y_pos = np.arange(len(task_keys))

    ax1.barh(y_pos, see_pcts, height=0.52, color="#00ACC1", label="Seeing Energy (Prefill / Vision)")
    ax1.barh(y_pos, speak_pcts, height=0.52, left=see_pcts, color="#FB8C00", label="Speaking Energy (Decode / Text)")
    ax1.axvline(50, color="#D32F2F", linestyle="--", linewidth=2.0, label="50% Crossover Boundary")

    for idx, (sp, spp) in enumerate(zip(see_pcts, speak_pcts)):
        if sp > 12:
            ax1.text(sp / 2, idx, f"{sp:.1f}%", va="center", ha="center", color="white", fontweight="bold", fontsize=10)
        if spp > 12:
            ax1.text(sp + spp / 2, idx, f"{spp:.1f}%", va="center", ha="center", color="white", fontweight="bold", fontsize=10)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(task_labels, fontsize=11, fontweight="bold")
    ax1.set_xlim(0, 100)
    ax1.set_xlabel(f"Energy Share (%) at {res_default} Resolution", fontsize=11, fontweight="semibold")
    ax1.set_title("The 'Seeing vs. Speaking' Phase Transition", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="lower right")

    # 2. Pareto Curves
    colors = {"DocVQA": "#1E88E5", "ChartQA": "#43A047", "POPE": "#8E24AA"}
    for t in [k for k in ["DocVQA", "ChartQA", "POPE"] if k in summary]:
        res_list = list(summary[t].keys())
        energies = [summary[t][r]["tot_j"] for r in res_list]
        accs = [summary[t][r]["accuracy"] for r in res_list]
        ax2.plot(energies, accs, marker="o", markersize=8, linewidth=2.2, color=colors.get(t, "#333333"), label=t)
        for e, a, r in zip(energies, accs, res_list):
            ax2.annotate(f"{r.split('x')[0]}px", (e, a), textcoords="offset points", xytext=(8, -4), fontsize=8.5, fontweight="bold")

    ax2.set_xlabel("Mean Inference Energy per Query (Joules)", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Task Accuracy (%)", fontsize=11, fontweight="semibold")
    ax2.set_title("Energy-Accuracy Pareto Frontiers across Resolutions", fontsize=13, fontweight="bold", pad=12)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved publication figure to '{output_path}'")


# -------------------------------------------------------------
# Main Runner
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GreenVLM Multi-Task Energy Benchmark")
    parser.add_argument("--model", type=str, default="moondream:latest", help="VLM name in Ollama")
    parser.add_argument("--tasks", nargs="+", default=["DocVQA", "ChartQA", "POPE", "COCO"], help="Tasks to benchmark")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[224, 448, 896], help="Square resolutions (e.g. 224 448 896)")
    parser.add_argument("--data-dir", type=str, default="data/multitask_samples", help="Path to sample images and metadata")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save JSON logs and figures")
    parser.add_argument("--limit", type=int, default=0, help="Max samples per task (0 = all). Use for a quick pilot.")
    parser.add_argument("--backend", type=str, default="ollama", choices=["ollama", "hf"],
                        help="Inference backend. 'ollama' (default) or 'hf' for HuggingFace models like InternVL.")
    parser.add_argument("--quant", type=str, default="4bit", choices=["4bit", "8bit", "bf16"],
                        help="Quantization for the HF backend (default 4bit).")
    args = parser.parse_args()

    # Unified query dispatch so the energy tracker, evaluators, checkpointing and
    # plotting are identical across backends.
    if args.backend == "hf":
        from internvl_backend import hf_query
        def run_query(prompt, img_b64, max_tokens):
            return hf_query(args.model, prompt, img_b64, max_tokens=max_tokens, quant=args.quant)
    else:
        def run_query(prompt, img_b64, max_tokens):
            return query_model(args.model, prompt, img_b64, max_tokens=max_tokens)

    os.makedirs(args.output_dir, exist_ok=True)
    res_tuples = [(r, r) for r in args.resolutions]

    meta_candidates = [
        os.path.join(args.data_dir, "metadata.json"),
        os.path.join(args.data_dir, "multitask_samples_metadata.json"),
        "multitask_samples/multitask_samples_metadata.json",
    ]
    meta_file = next((c for c in meta_candidates if os.path.exists(c)), None)
    if meta_file is None:
        print(f"[!] Metadata not found. Tried: {meta_candidates}. Please run 'python download_datasets.py' first.")
        sys.exit(1)

    # Resolve image paths relative to the data dir's parent (metadata stores paths
    # like 'multitask_samples/docvqa/doc_00.png' relative to 'data/').
    data_base = os.path.dirname(os.path.abspath(args.data_dir))

    with open(meta_file, "r") as f:
        all_meta = json.load(f)

    print("=" * 86)
    print("  GreenVLM: The Illusion of 'Free' Vision in Real-World Multimodal Tasks")
    print("=" * 86)
    print(f"Model       : {args.model}")
    print(f"Tasks       : {args.tasks}")
    print(f"Resolutions : {[f'{r[0]}x{r[1]}' for r in res_tuples]}")
    print("=" * 86)

    task_tokens = {"DocVQA": 20, "ChartQA": 15, "POPE": 5, "COCO": 45}
    full_log = []
    summary = {t: {} for t in args.tasks}

    # ------------------------------------------------------------------
    # Checkpoint / resume. Every completed inference is appended to
    # <output-dir>/checkpoint.jsonl immediately (flushed + fsynced). Re-running
    # the exact same command skips already-finished (task, resolution, sample)
    # items and resumes where it left off after an interruption.
    # ------------------------------------------------------------------
    ckpt_path = os.path.join(args.output_dir, "checkpoint.jsonl")
    done = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as cf:
            for line in cf:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a half-written final line from a hard kill
                done[(rec["task"], rec["resolution"], rec["sample_id"])] = rec
        print(f"[*] Resume: loaded {len(done)} completed inferences from '{ckpt_path}'.")
    ckpt_file = open(ckpt_path, "a")

    # Lazily load the model into GPU memory on the first *real* inference so the
    # cold-load cost is never charged to a benchmarked sample, and a fully-cached
    # resume run does not reload the model at all.
    warmed = [False]
    def ensure_warm(img_b64_warm):
        if warmed[0]:
            return
        try:
            print("[*] Warming up model (loading into GPU memory)...")
            run_query("Describe this image.", img_b64_warm, max_tokens=5)
            print("[✓] Warmup complete.")
        except Exception as e:
            print(f"[!] Warmup skipped: {e}")
        warmed[0] = True

    for task in args.tasks:
        samples = all_meta.get(task, [])
        if not samples:
            continue
        if args.limit > 0:
            samples = samples[:args.limit]
        print(f"\n>>> Running Benchmark on Task: {task.upper()} ({len(samples)} samples)...")

        for res in res_tuples:
            res_str = f"{res[0]}x{res[1]}"
            acc_list, see_t_list, speak_t_list, see_j_list, speak_j_list, tot_j_list, out_tok_list = [], [], [], [], [], [], []

            for idx, s in enumerate(samples, start=1):
                key = (task, res_str, s["id"])
                if key in done:
                    rec = done[key]
                    acc_list.append(rec["acc"]); see_t_list.append(rec["t_see"]); speak_t_list.append(rec["t_speak"])
                    see_j_list.append(rec["see_j"]); speak_j_list.append(rec["speak_j"]); tot_j_list.append(rec["tot_j"])
                    out_tok_list.append(rec["out_tok"]); full_log.append(rec)
                    continue

                q = s["question"]
                gts = s["ground_truth"]
                img_path = s["image_path"]
                if not os.path.exists(img_path):
                    cand = os.path.join(data_base, img_path)
                    if os.path.exists(cand):
                        img_path = cand

                if task == "DocVQA":
                    prompt = f"Question: {q}\nAnswer directly and concisely using words from the document."
                elif task == "ChartQA":
                    prompt = f"Question: {q}\nAnswer concisely with the exact value or category from the chart."
                elif task == "POPE":
                    prompt = f"Question: {q}\nAnswer with exactly one word: 'yes' or 'no'."
                else:
                    prompt = "Describe this image concisely in 2 sentences."

                img_b64 = resize_image_b64(img_path, res)
                ensure_warm(img_b64)

                tracker = HardwareEnergyTracker(interval_sec=0.02)
                tracker.start()

                try:
                    resp = run_query(prompt, img_b64, max_tokens=task_tokens.get(task, 25))
                except Exception as e:
                    tracker.stop()
                    print(f"    [Error] Sample {idx} failed: {e}")
                    continue

                p_stats = tracker.stop()

                t_see = resp.get("prompt_eval_duration", 0) / 1e9
                t_speak = resp.get("eval_duration", 0) / 1e9
                tot_j = p_stats["energy_joules"]
                t_act = max(t_see + t_speak, 0.001)
                rho_see = t_see / t_act
                see_j = tot_j * rho_see
                speak_j = tot_j * (1.0 - rho_see)

                ans = resp.get("response", "").strip()
                out_tok = resp.get("eval_count", 0)

                if task == "DocVQA":
                    acc = compute_anls(ans, gts)
                elif task == "ChartQA":
                    acc = eval_chartqa(ans, gts)
                elif task == "POPE":
                    acc = eval_pope(ans, gts[0] if gts else "no")
                else:
                    acc = float(out_tok)

                rec = {
                    "task": task, "sample_id": s["id"], "resolution": res_str,
                    "prediction": ans, "score": acc, "acc": acc, "out_tok": out_tok,
                    "see_j": see_j, "speak_j": speak_j, "tot_j": tot_j,
                    "t_see": t_see, "t_speak": t_speak
                }
                # Persist immediately so an interruption loses at most this one sample.
                ckpt_file.write(json.dumps(rec) + "\n"); ckpt_file.flush(); os.fsync(ckpt_file.fileno())
                done[key] = rec

                acc_list.append(acc)
                see_t_list.append(t_see)
                speak_t_list.append(t_speak)
                see_j_list.append(see_j)
                speak_j_list.append(speak_j)
                tot_j_list.append(tot_j)
                out_tok_list.append(out_tok)
                full_log.append(rec)

            mean_acc = float(np.mean(acc_list)) * (1.0 if task == "COCO" else 100.0)
            mean_tot_j = float(np.mean(tot_j_list))
            mean_see_j = float(np.mean(see_j_list))
            mean_speak_j = float(np.mean(speak_j_list))
            rho_pre = (mean_see_j / mean_tot_j * 100.0) if mean_tot_j > 0 else 0.0

            summary[task][res_str] = {
                "accuracy": mean_acc, "see_j": mean_see_j, "speak_j": mean_speak_j,
                "tot_j": mean_tot_j, "see_t": float(np.mean(see_t_list)),
                "speak_t": float(np.mean(speak_t_list)), "out_toks": float(np.mean(out_tok_list)),
                "rho_pre": rho_pre
            }
            metric_label = f"{mean_acc:.1f} toks" if task == "COCO" else f"{mean_acc:.2f}%"
            print(f"  [{res_str}] {task:<8} | Acc: {metric_label:<10} | See: {mean_see_j:5.1f}J ({rho_pre:4.1f}%) | Speak: {mean_speak_j:4.1f}J | Total: {mean_tot_j:5.1f}J")

    ckpt_file.close()

    # Save Results
    json_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "details": full_log}, f, indent=2)
    print(f"\n[✓] Saved JSON results to '{json_path}'")

    fig_path = os.path.join(args.output_dir, "phase_transition_and_pareto.png")
    generate_plots(summary, fig_path)

if __name__ == "__main__":
    main()
