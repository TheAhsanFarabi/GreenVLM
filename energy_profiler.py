"""
GreenVLM: Hardware Energy Profiler.
Supports:
1. Intel/AMD CPU Hardware Registers (Linux RAPL)
2. NVIDIA GPU Hardware Telemetry (pynvml)
3. Dynamic CPU / Platform Power Modeling (Fallback)
"""

import os
import time
import threading
from typing import Dict, Any, Optional

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

RAPL_PATH = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"

def can_read_rapl() -> bool:
    if os.path.exists(RAPL_PATH):
        try:
            with open(RAPL_PATH, "r") as f:
                int(f.read().strip())
            return True
        except (PermissionError, ValueError):
            return False
    return False

def read_rapl_joules() -> Optional[float]:
    if not os.path.exists(RAPL_PATH):
        return None
    try:
        with open(RAPL_PATH, "r") as f:
            return int(f.read().strip()) / 1_000_000.0
    except Exception:
        return None


class HardwareEnergyTracker:
    """
    Continuous hardware power monitor.
    Samples every 20-50ms and integrates Energy = sum(Power * dt).
    Automatically leverages NVIDIA NVML on GPU workstations.
    """
    def __init__(self, interval_sec: float = 0.02, default_gpu_index: int = 0):
        self.interval = interval_sec
        self.gpu_idx = default_gpu_index
        self.running = False
        self.thread = None
        self.samples = []
        self.energy_joules = 0.0
        self.rapl_start = None
        self.rapl_end = None
        
        self.gpu_handle = None
        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_idx)
            except Exception:
                self.gpu_handle = None

    def _loop(self):
        last_time = time.time()
        while self.running:
            time.sleep(self.interval)
            now = time.time()
            dt = now - last_time
            last_time = now

            inst_power = None
            # 1. Try GPU NVML
            if self.gpu_handle:
                try:
                    milliwatts = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                    inst_power = milliwatts / 1000.0
                except Exception:
                    pass

            # 2. Try CPU Model
            if inst_power is None:
                if HAS_PSUTIL:
                    cpu_pct = psutil.cpu_percent(interval=None) / 100.0
                    idle_w = 4.0
                    max_w = 45.0
                    inst_power = idle_w + (max_w - idle_w) * cpu_pct
                else:
                    inst_power = 15.0

            self.samples.append(inst_power)
            self.energy_joules += inst_power * dt

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

        avg_power = (sum(self.samples) / len(self.samples)) if self.samples else 15.0

        if self.gpu_handle:
            method = "NVIDIA NVML Hardware Rail"
            joules = self.energy_joules
        elif self.rapl_start is not None and self.rapl_end is not None and self.rapl_end >= self.rapl_start:
            joules = self.rapl_end - self.rapl_start
            method = "Intel RAPL Hardware Register"
        else:
            joules = self.energy_joules
            method = "CPU Dynamic Power Model"

        return {
            "energy_joules": max(joules, 0.001),
            "avg_power_watts": avg_power,
            "method": method
        }
