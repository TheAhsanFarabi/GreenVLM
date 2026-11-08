#!/usr/bin/env python3
"""
GreenVLM Quick Experiment:
Testing Vision-Language Energy across 3 Image Resolutions using Ollama.

Disentangles:
- Seeing (Visual Prefill: ViT + Projection + Prompt Ingestion)
- Speaking (Autoregressive Text Generation / Decoding)
- Total Energy (Joules)
"""

import os
import sys
import time
import json
import base64
import threading
from io import BytesIO
from typing import Dict, Any, Optional

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Pillow is required. Install via: pip install Pillow")
    sys.exit(1)

try:
    import urllib.request
except ImportError:
    pass

try:
    import psutil
except ImportError:
    psutil = None


# -------------------------------------------------------------
# 1. RAPL Hardware Energy Reader (Linux Intel / AMD)
# -------------------------------------------------------------
RAPL_PATH = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"

def can_read_rapl() -> bool:
    if os.path.exists(RAPL_PATH):
        try:
            with open(RAPL_PATH, "r") as f:
                int(f.read().strip())
            return True
        except PermissionError:
            return False
    return False

def read_rapl_joules() -> Optional[float]:
    """Reads current cumulative energy in Joules from Intel RAPL register."""
    if not os.path.exists(RAPL_PATH):
        return None
    try:
        with open(RAPL_PATH, "r") as f:
            microjoules = int(f.read().strip())
            return microjoules / 1_000_000.0
    except Exception:
        return None


# -------------------------------------------------------------
# 2. Dynamic Power Sampler (Fallback if RAPL is restricted)
# -------------------------------------------------------------
class BackgroundPowerTracker:
    """
    Samples CPU utilization and estimated power draw every 50ms.
    Integrates Energy = sum(Power * dt).
    """
    def __init__(self, interval_sec: float = 0.05, base_tdp_watts: float = 15.0, max_tdp_watts: float = 45.0):
        self.interval = interval_sec
        self.base_tdp = base_tdp_watts
        self.max_tdp = max_tdp_watts
        self.running = False
        self.thread = None
        self.energy_joules = 0.0
        self.samples = []
        self.rapl_start = None
        self.rapl_end = None

    def _loop(self):
        last_time = time.time()
        while self.running:
            time.sleep(self.interval)
            now = time.time()
            dt = now - last_time
            last_time = now

            if psutil:
                cpu_pct = psutil.cpu_percent(interval=None) / 100.0
                # Power model: Idle (~4W) + dynamic load up to max_tdp
                idle_w = 4.0
                est_power = idle_w + (self.max_tdp - idle_w) * cpu_pct
            else:
                est_power = self.base_tdp
            
            self.samples.append(est_power)
            self.energy_joules += est_power * dt

    def start(self):
        self.running = True
        self.energy_joules = 0.0
        self.samples = []
        self.rapl_start = read_rapl_joules()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> Dict[str, Any]:
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        self.rapl_end = read_rapl_joules()

        avg_power = (sum(self.samples) / len(self.samples)) if self.samples else self.base_tdp

        # Prefer direct hardware RAPL counter if accessible
        if self.rapl_start is not None and self.rapl_end is not None:
            joules = self.rapl_end - self.rapl_start
            method = "Hardware Register (Intel RAPL)"
        else:
            joules = self.energy_joules
            method = "Dynamic CPU Power Model"

        return {
            "energy_joules": max(joules, 0.001),
            "avg_power_watts": avg_power,
            "method": method
        }


# -------------------------------------------------------------
# 3. Image Preparation & Resizing
# -------------------------------------------------------------
def create_sample_test_image(save_path: str = "sample_test.png"):
    """Creates a high-contrast test image containing text, charts, and shapes."""
    img = Image.new("RGB", (1024, 1024), color=(245, 247, 250))
    draw = ImageDraw.Draw(img)

    # Header banner
    draw.rectangle([(40, 40), (984, 140)], fill=(33, 150, 243))
    draw.text((60, 65), "GREEN-VLM ENERGY BENCHMARK 2026", fill=(255, 255, 255))
    draw.text((60, 100), "Project ID: #9482 | Task: Fine-Grained Document QA", fill=(230, 240, 255))

    # Synthetic chart / table
    draw.rectangle([(50, 180), (500, 600)], outline=(100, 100, 100), width=3)
    draw.text((70, 200), "DATA TABLE:", fill=(0, 0, 0))
    items = [
        ("Resolution", "Tokens", "Energy"),
        ("224 x 224", "196", "12.4 J"),
        ("448 x 448", "784", "24.1 J"),
        ("896 x 896", "3136", "68.9 J"),
    ]
    y = 240
    for row in items:
        draw.text((70, y), f"{row[0]:<14} | {row[1]:<8} | {row[2]}", fill=(30, 30, 30))
        y += 45

    # Visual shapes
    draw.ellipse([(600, 220), (850, 470)], fill=(255, 87, 34), outline=(180, 40, 10), width=4)
    draw.polygon([(725, 550), (600, 750), (850, 750)], fill=(76, 175, 80), outline=(40, 120, 40), width=3)

    # Footer query text
    draw.rectangle([(50, 820), (984, 960)], fill=(220, 230, 242))
    draw.text((70, 850), "TARGET QUESTION: What is the energy value listed for 896 x 896?", fill=(10, 10, 10))
    draw.text((70, 900), "PIN CODE FOR VERIFICATION: [CODE-7721]", fill=(180, 0, 0))

    img.save(save_path)
    print(f"[*] Generated synthetic test image: {save_path} (1024x1024)")
    return save_path

def resize_and_encode_image(image_path: str, size: tuple) -> tuple:
    """Resizes image to target (W, H) and returns (base64_str, byte_size)."""
    with Image.open(image_path) as img:
        img_resized = img.resize(size, Image.Resampling.LANCZOS)
        buffered = BytesIO()
        img_resized.save(buffered, format="PNG")
        raw_bytes = buffered.getvalue()
        b64 = base64.b64encode(raw_bytes).decode("utf-8")
        return b64, len(raw_bytes)


# -------------------------------------------------------------
# 4. Ollama Client & Inference Call
# -------------------------------------------------------------
def query_ollama(model: str, prompt: str, image_b64: str, max_tokens: int = 40) -> Dict[str, Any]:
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
            "temperature": 0.2,
            "num_predict": max_tokens
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(req, timeout=180) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        # Standardize response content field
        if "message" in res and "content" in res["message"]:
            res["response"] = res["message"]["content"]
        return res


# -------------------------------------------------------------
# 5. Main Experiment Runner
# -------------------------------------------------------------
def run_experiment(
    image_path: Optional[str] = None,
    model_name: str = "moondream",
    prompt: str = "Describe the contents and layout of this image concisely in 2 sentences.",
    resolutions = [(224, 224), (448, 448), (896, 896)]
):
    print("=" * 78)
    print("      GREEN-VLM: 'SEEING VS. SPEAKING' ENERGY EXPERIMENT")
    print("=" * 78)
    print(f"Model       : {model_name}")
    print(f"Prompt      : \"{prompt}\"")
    print(f"RAPL Access : {'Enabled (Hardware register)' if can_read_rapl() else 'Restricted (Using CPU power model)'}")
    if not can_read_rapl():
        print("  [Tip] To enable microjoule hardware precision, run:")
        print("        sudo chmod -R a+r /sys/class/powercap/intel-rapl/")
    print("=" * 78)

    # 1. Image check
    if not image_path or not os.path.exists(image_path):
        image_path = create_sample_test_image("sample_test.png")

    results = []

    for idx, res in enumerate(resolutions, start=1):
        res_str = f"{res[0]}x{res[1]}"
        print(f"\n[{idx}/3] Testing Resolution: {res_str} ...")

        # Prepare resized image
        img_b64, byte_len = resize_and_encode_image(image_path, res)
        print(f"     Image payload: {byte_len / 1024:.1f} KB")

        # Start energy tracker
        tracker = BackgroundPowerTracker(interval_sec=0.02)
        tracker.start()

        # Run inference
        try:
            resp = query_ollama(model=model_name, prompt=prompt, image_b64=img_b64)
        except Exception as e:
            tracker.stop()
            print(f"     [Error] Ollama inference failed: {e}")
            print(f"     Make sure '{model_name}' is installed (`ollama pull {model_name}`) and Ollama is running.")
            return

        # Stop energy tracker
        power_stats = tracker.stop()

        # Extract timing and token counts from Ollama
        # Nanoseconds -> seconds
        prompt_eval_sec = resp.get("prompt_eval_duration", 0) / 1e9
        eval_sec = resp.get("eval_duration", 0) / 1e9
        total_time_sec = resp.get("total_duration", 0) / 1e9
        if total_time_sec <= 0:
            total_time_sec = prompt_eval_sec + eval_sec

        in_tokens = resp.get("prompt_eval_count", 0)
        out_tokens = resp.get("eval_count", 0)
        answer = resp.get("response", "").strip()

        # Compute Energy Decomposition
        total_joules = power_stats["energy_joules"]
        avg_power = power_stats["avg_power_watts"]

        # If using Zhan et al.'s model: E = P_bar * t
        # Prefill Share rho_pre = t_prefill / (t_prefill + t_decode)
        active_time = max(prompt_eval_sec + eval_sec, 0.001)
        rho_pre = prompt_eval_sec / active_time
        rho_dec = eval_sec / active_time

        seeing_joules = total_joules * rho_pre
        speaking_joules = total_joules * rho_dec

        results.append({
            "resolution": res_str,
            "in_tokens": in_tokens,
            "out_tokens": out_tokens,
            "see_time": prompt_eval_sec,
            "speak_time": eval_sec,
            "total_time": total_time_sec,
            "see_joules": seeing_joules,
            "speak_joules": speaking_joules,
            "total_joules": total_joules,
            "rho_pre": rho_pre * 100.0,
            "rho_dec": rho_dec * 100.0,
            "answer": answer
        })

        print(f"     Response: \"{answer[:80]}{'...' if len(answer)>80 else ''}\"")
        print(f"     Seeing Time  : {prompt_eval_sec:6.2f}s | Speaking Time: {eval_sec:6.2f}s")
        print(f"     Seeing Share : {rho_pre*100:5.1f}%  | Speaking Share: {rho_dec*100:5.1f}%")
        print(f"     Total Energy : {total_joules:6.2f} Joules ({power_stats['method']})")

    # ---------------------------------------------------------
    # Final Comparative Summary Table
    # ---------------------------------------------------------
    print("\n" + "=" * 90)
    print("                     FINAL EXPERIMENTAL SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Resolution':<12} | {'Tokens (In/Out)':<15} | {'Seeing (s)':<10} | {'Speaking (s)':<12} | {'See Energy':<12} | {'Speak Energy':<13} | {'Total (J)':<9} | {'Seeing %':<8}")
    print("-" * 90)
    for r in results:
        tok_str = f"{r['in_tokens']} / {r['out_tokens']}"
        print(f"{r['resolution']:<12} | {tok_str:<15} | {r['see_time']:<10.2f} | {r['speak_time']:<12.2f} | {r['see_joules']:<10.2f} J | {r['speak_joules']:<11.2f} J | {r['total_joules']:<8.2f}J | {r['rho_pre']:<7.1f}%")
    print("=" * 90)

    # ---------------------------------------------------------
    # Analysis & Crossover Insights
    # ---------------------------------------------------------
    print("\n[Analysis & Findings vs. Zhan et al.]")
    low_rho = results[0]["rho_pre"]
    high_rho = results[-1]["rho_pre"]
    print(f"1. Seeing Share Transition: {low_rho:.1f}% (at {results[0]['resolution']}) -> {high_rho:.1f}% (at {results[-1]['resolution']})")
    if high_rho > 50.0:
        print("   >>> CROSSOVER CONFIRMED! At high resolution, SEEING overtook SPEAKING as the dominant energy cost!")
    else:
        print("   >>> Decode remained dominant, but prefill share increased as resolution scaled.")

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    img_arg = sys.argv[1] if len(sys.argv) > 1 else None
    model_arg = sys.argv[2] if len(sys.argv) > 2 else "moondream"
    run_experiment(image_path=img_arg, model_name=model_arg)
