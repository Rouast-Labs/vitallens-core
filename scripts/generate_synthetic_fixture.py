"""Generates a synthetic 3-minute PPG fixture with known beat times.

Real fixtures are at most about 2 minutes long, too short for LF/HF and SD1/SD2. This one
builds a pulse waveform from a beat sequence whose intervals carry LF (0.1 Hz) and HF
(0.25 Hz) modulation plus noise. HRV references come from NeuroKit2 on the exact beat
times, independently of the core, so the test covers the whole path from waveform to metric.
The stress index, which NeuroKit2 does not provide, uses the reference implementation in
prepare_test_data.py.

Usage: python scripts/generate_synthetic_fixture.py [--out tests/fixtures/synthetic_3min.json]
"""
import argparse
import json
import numpy as np
import neurokit2 as nk
from prepare_test_data import stress_index_windowed_reference

FS = 30.0
DURATION_S = 180.0
CONFIDENCE = 0.95
SEED = 0


def beat_times(rng):
  """Beat times in seconds from intervals with LF and HF modulation."""
  times = [0.5]
  while times[-1] < DURATION_S:
    t = times[-1]
    rr_ms = (900.0
             + 45.0 * np.sin(2 * np.pi * 0.10 * t)
             + 30.0 * np.sin(2 * np.pi * 0.25 * t + 1.0)
             + rng.normal(0.0, 8.0))
    times.append(t + rr_ms / 1000.0)
  return np.array(times[:-1])


def ppg_waveform(beats, rng):
  """Systolic and smaller diastolic wave per beat, with baseline wander and noise."""
  t = np.arange(int(DURATION_S * FS)) / FS
  x = np.zeros_like(t)
  for tb in beats:
    x += np.exp(-0.5 * ((t - tb) / 0.08) ** 2)
    x += 0.35 * np.exp(-0.5 * ((t - tb - 0.30) / 0.10) ** 2)
  x += 0.10 * np.sin(2 * np.pi * 0.05 * t)
  x += rng.normal(0.0, 0.03, len(t))
  return (x - x.mean()) / x.std()


def references(beats):
  peaks = np.round(beats * 1000).astype(int)
  td = nk.hrv_time(peaks, sampling_rate=1000)
  fd = nk.hrv_frequency(peaks, sampling_rate=1000, psd_method="fft")
  nl = nk.hrv_nonlinear(peaks, sampling_rate=1000)
  return {
    "heart_rate": (60000.0 / np.mean(np.diff(beats) * 1000), "bpm"),
    "hrv_sdnn": (td["HRV_SDNN"].values[0], "ms"),
    "hrv_rmssd": (td["HRV_RMSSD"].values[0], "ms"),
    "hrv_pnn50": (td["HRV_pNN50"].values[0], "%"),
    "hrv_lfhf": (fd["HRV_LFHF"].values[0], "ratio"),
    "hrv_sd1sd2": (nl["HRV_SD1SD2"].values[0], "ratio"),
    "stress_index": (stress_index_windowed_reference(np.diff(beats) * 1000, beats[1:]), "SI"),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", default="tests/fixtures/synthetic_3min.json")
  args = parser.parse_args()

  rng = np.random.default_rng(SEED)
  beats = beat_times(rng)
  x = ppg_waveform(beats, rng)
  peak_indices = [int(round(tb * FS)) for tb in beats if round(tb * FS) < len(x)]

  vitals = {}
  for key, (value, unit) in references(beats).items():
    vitals[key] = {
      "value": round(float(value), 2),
      "unit": unit,
      "confidence": CONFIDENCE,
      "note": "Synthetic (NeuroKit2 from known beat times)",
    }

  fixture = {
    "waveforms": {
      "ppg_waveform": {
        "data": [round(float(v), 4) for v in x],
        "unit": "unitless",
        "confidence": [CONFIDENCE] * len(x),
        "note": "Synthetic pulse waveform",
        "peak_indices": peak_indices,
      }
    },
    "vitals": vitals,
    "fps": FS,
    "n": len(x),
    "source": f"scripts/generate_synthetic_fixture.py (seed {SEED})",
  }
  with open(args.out, "w") as f:
    json.dump(fixture, f, indent=4)
  print({k: v["value"] for k, v in vitals.items()})


if __name__ == "__main__":
  main()
