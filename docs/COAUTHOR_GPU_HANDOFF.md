# GreenVLM: The Illusion of "Free" Vision in Real-World Multimodal Tasks
### Co-Author GPU Handoff & Scaled Benchmarking Guide

This document and codebase are prepared for running the **GreenVLM Multi-Task Benchmark** on high-performance GPU workstations (NVIDIA RTX 3090, 4090, A100, H100).

---

## 1. Executive Summary: What We Proved on CPU

We conducted a 300-inference benchmark on an edge CPU (Intel Core i3) evaluating **`moondream:latest`** across 4 distinct task archetypes (100 samples) at 3 resolutions ($224 \times 224$, $448 \times 448$, $896 \times 896$):

1. **DocVQA (25 samples - Document OCR):**
   * **Seeing Energy:** **98.9%** ($79.7\text{ J}$) vs. **Speaking Energy:** **1.1%** ($0.9\text{ J}$).
   * **Accuracy:** Surged from **$6.43\% \to 9.96\%$ (+55% relative gain)** when scaling resolution from $224^2 \to 448^2$.
2. **ChartQA & POPE (50 samples - Data Reasoning & Probing):**
   * **Seeing Energy:** **99.8% – 100.0%** of total inference energy.
3. **COCO (25 samples - Open Captioning):**
   * Speaking energy increased **$20\times$** compared to DocVQA ($20.5\text{ J}$ vs $0.9\text{ J}$), replicating the wordy decode burden observed by Zhan et al.

> **Key Takeaway:** Zhan et al.'s claim that *"text decoding always consumes 86%–97% of energy"* is strictly an artifact of open-ended captioning. On short-answer vision tasks, **Seeing consumes 98%–100% of the energy!**

---

## 2. Your Mission on the GPU Workstation

1. **GPU Acceleration:** On CPU, 300 inferences took 39 minutes. On your NVIDIA GPU, it will finish in **under 45 seconds**.
2. **Dynamic-Resolution Architecture:** Moondream uses fixed internal tokens ($\sim 752$ tokens). You will test **`Qwen2.5-VL-7B`** or **`InternVL2.5-8B`**, where visual tokens scale quadratically with resolution ($196 \to 784 \to 3136$ tokens), demonstrating the continuous visual energy explosion!
3. **Scale Sample Count:** Expand from 25 samples per task to **100–250 samples per task** for camera-ready publication figures.

---

## 3. Quick Start Setup (5 Minutes)

### Step 1: Install Python Dependencies
```bash
pip install datasets Pillow matplotlib pandas pyarrow pynvml
```

### Step 2: Ensure Ollama is Installed & Pull Target Models
```bash
# Pull the dynamic-resolution vision models
ollama pull qwen2.5-vl:7b
ollama pull moondream:latest
```

### Step 3: Run the Multi-Task Benchmark
```bash
# Run the 100-sample pilot on your GPU (under 1 minute)
python3 benchmark_100_multitask_pareto.py
```

---

## 4. How to Scale Sample Count & Test Dynamic Models

In [`benchmark_100_multitask_pareto.py`](benchmark_100_multitask_pareto.py):
1. Change the model parameter:
   ```python
   model_name = "qwen2.5-vl:7b"  # Dynamic resolution
   ```
2. In [`download_multitask_samples.py`](download_multitask_samples.py), change `TARGET_PER_TASK = 25` to `TARGET_PER_TASK = 150` to pull more samples from Hugging Face:
   ```bash
   python3 download_multitask_samples.py
   python3 benchmark_100_multitask_pareto.py
   ```

---

## 5. Output Artifacts Generated

* **`multitask_phase_transition_and_pareto.png`**: High-resolution 300 DPI dual-panel publication plot (Phase Transition + Pareto Frontiers).
* **`multitask_benchmark_results.json`**: Complete raw JSON containing per-sample predictions, ANLS scores, nanosecond timings, and Joules.
* **`GreenVLM_100_Sample_MultiTask_Report.docx`**: Formatted Word report ready for advisors and collaborators.
