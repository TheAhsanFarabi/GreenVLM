#!/usr/bin/env python3
"""
GreenVLM: Multi-Task Dataset Downloader & Cache.
Pulls standardized benchmarks from Hugging Face:
- DocVQA (Document Intelligence)
- ChartQA (Visual Data Reasoning)
- POPE (Object Probing / Hallucination)
- COCO (Image Captioning)

Usage:
    python download_datasets.py --samples 50 --output data/multitask_samples
"""

import os
import sys
import json
import argparse
import datasets
from PIL import Image

def download_task_samples(task: str, target_count: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    task_dir = os.path.join(output_dir, task.lower())
    os.makedirs(task_dir, exist_ok=True)

    print(f"\n[*] Pulling {target_count} samples for {task}...")
    samples = []

    if task == "DocVQA":
        ds = datasets.load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation", streaming=True)
        count = 0
        for row in ds:
            if count >= target_count:
                break
            img, q, ans = row.get("image"), row.get("question"), row.get("answers", [])
            if not img or not q or not ans:
                continue
            path = os.path.join(task_dir, f"doc_{count:03d}.png")
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(path)
            samples.append({"id": count, "task": "DocVQA", "image_path": path, "question": q, "ground_truth": ans})
            count += 1
            if count % 10 == 0 or count == target_count:
                print(f"    [DocVQA] {count}/{target_count} cached")

    elif task == "ChartQA":
        ds = datasets.load_dataset("HuggingFaceM4/ChartQA", split="test", streaming=True)
        count = 0
        for row in ds:
            if count >= target_count:
                break
            img, q, ans = row.get("image"), row.get("query"), row.get("label", [])
            if not img or not q or not ans:
                continue
            path = os.path.join(task_dir, f"chart_{count:03d}.png")
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(path)
            samples.append({"id": count, "task": "ChartQA", "image_path": path, "question": q, "ground_truth": ans})
            count += 1
            if count % 10 == 0 or count == target_count:
                print(f"    [ChartQA] {count}/{target_count} cached")

    elif task == "POPE":
        ds = datasets.load_dataset("lmms-lab/POPE", split="test", streaming=True)
        count = 0
        for row in ds:
            if count >= target_count:
                break
            img, q, ans = row.get("image"), row.get("question"), row.get("answer")
            if not img or not q or not ans:
                continue
            path = os.path.join(task_dir, f"pope_{count:03d}.png")
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(path)
            samples.append({"id": count, "task": "POPE", "image_path": path, "question": q, "ground_truth": [str(ans).strip().lower()]})
            count += 1
            if count % 10 == 0 or count == target_count:
                print(f"    [POPE] {count}/{target_count} cached")

    elif task == "COCO":
        ds = datasets.load_dataset("lmms-lab/COCO-Caption", split="val", streaming=True)
        count = 0
        for row in ds:
            if count >= target_count:
                break
            img = row.get("image")
            if not img:
                continue
            path = os.path.join(task_dir, f"coco_{count:03d}.png")
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(path)
            samples.append({"id": count, "task": "COCO", "image_path": path, "question": "Describe this image in detail.", "ground_truth": []})
            count += 1
            if count % 10 == 0 or count == target_count:
                print(f"    [COCO] {count}/{target_count} cached")

    return samples

def main():
    parser = argparse.ArgumentParser(description="Download benchmarks for GreenVLM")
    parser.add_argument("--samples", type=int, default=25, help="Number of samples per task (default: 25)")
    parser.add_argument("--tasks", nargs="+", default=["DocVQA", "ChartQA", "POPE", "COCO"], help="Tasks to pull")
    parser.add_argument("--output", type=str, default="data/multitask_samples", help="Output directory")
    args = parser.parse_args()

    all_meta = {}
    for t in args.tasks:
        samples = download_task_samples(t, args.samples, args.output)
        all_meta[t] = samples

    meta_file = os.path.join(args.output, "metadata.json")
    with open(meta_file, "w") as f:
        json.dump(all_meta, f, indent=2)

    print(f"\n[✓] Successfully cached datasets to '{args.output}/'")

if __name__ == "__main__":
    main()
    # The `datasets`/`pyarrow`/`aiohttp` streaming backends spin up background
    # threads that are not always joined cleanly at interpreter shutdown, which
    # can trigger a benign-but-alarming "Fatal Python error: PyGILState_Release"
    # (shown by Ubuntu's apport as "python3.12 unexpectedly stopped") *after* the
    # data is already written. Flush and hard-exit to skip that buggy finalizer.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
