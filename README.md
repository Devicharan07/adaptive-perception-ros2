# State-Conditioned Adaptive Allocation (SCAA) for Fiducial Perception

ROS 2 perception package implementing resource-aware, state-conditioned adaptive allocation for square fiducial marker (ArUco) detection and relative pose tracking.

---

## Overview

Standard fiducial marker pipelines process every incoming camera frame at full native resolution, expending constant computational effort regardless of target distance, velocity, or scene dynamics.

**SCAA (State-Conditioned Adaptive Allocation)** is an adaptive perception controller that conditionally allocates compute:
- **Two-State Finite State Machine (FSM):** Transitions between global native-scale search (`RECOVERING` @ 1.0x full frame) and localized tracking (`LOCKED` @ adaptive ROI).
- **Temporal Reliability Estimation (EWMA):** Tracks multi-frame detection validity (alpha = 0.15, ~6 frames effective memory) instead of noisy single-frame confidence.
- **Dwell Hysteresis:** Enforces a 5-frame dwell requirement before downscaling to prevent rapid scale oscillation (scale flapping).
- **Marker-Size Resolution Gate:** Restricts aggressive 0.25x downscaling when apparent marker size falls below 120 px, preventing decode failure.
- **Asymmetric Recovery:** Fast upscale on detection miss (tolerates 1 isolated miss at 0.25x before stepping up to 0.6x; immediate step-up to 1.0x on any subsequent miss).

---

## Benchmark Highlights

Evaluated across **144 independent real-world camera trials** (~189.5k frames) across three operating regimes (hover/drift, close approach/retreat, and dynamic field-of-view exits):

- **Workload Reduction:** Cuts pixel workload by **81.7%** relative to the Full-Frame Baseline (p < 0.0001).
- **Perception Latency:** Reduces mean end-to-end latency from **8.78 ms -> 5.45 ms** (37.9% reduction, p < 0.0001, dz = -1.13).
- **Close Approach Regime:** SCAA achieves **97.36%** detection rate during scale transitions, outperforming unconstrained geometric allocation (94.16%, p = 0.0015) due to resolution gating.
- **Controller Stability:** Reduces scale transitions by **31.4%** overall compared to instantaneous geometric downscaling.

---

## Repository Structure

```text
.
|-- controller.py       # SCAA state machine, EWMA reliability, scale ladder, and gating logic
|-- detector.py         # OpenCV ArUco marker detection & corner extraction
|-- confidence.py       # Geometric confidence evaluator & subpixel refinement gating
|-- roi.py              # Region-of-Interest cropping, dynamic padding, and coordinate mapping
|-- pose.py             # IPPE / PnP 6-DOF relative camera-marker pose estimation
|-- profiler.py         # Per-frame execution profiler and latency telemetry logging
|-- main.py             # ROS 2 perception node runner (adaptive pipeline)
|-- baseline_main.py    # Reference baseline runners (Full-Frame & Kalman tracking)
`-- utils.py            # Mathematical transforms and visualization helpers
```

---

## Requirements & Software Stack

- **OS:** Ubuntu 22.04 LTS (Jammy Jellyfish)
- **Middleware:** ROS 2 Humble Hawksbill
- **Language & Libraries:** Python 3.10+, OpenCV 4.5.4+ (cv2.aruco), NumPy, SciPy

---

## Quickstart

### 1. Build within a ROS 2 Workspace

```bash
cd ~/ros2_ws/src
colcon build --packages-select adaptive_perception --symlink-install
source ~/ros2_ws/install/setup.bash
```

### 2. Run SCAA Perception Node

```bash
# Launch SCAA adaptive tracking with USB camera index 2
ros2 run adaptive_perception main --cam 2
```

### 3. Run Baselines for Comparison

```bash
# Full-Frame Baseline (FFB)
ros2 run adaptive_perception baseline_main --cam 2 --mode baseline

# Kalman ROI Tracker (KRT)
ros2 run adaptive_perception baseline_main --cam 2 --mode kalman
```

---

## Research & Publication

This repository provides the core perception implementation and real-time controller evaluated in the manuscript:  
**"State-Conditioned Adaptive Allocation: A Regime-Dependent Resource-Aware Perception Controller for Fiducial Marker Tracking"**  
*Author: Vannempalli Devi Charan (School of Computer Science, Manipal Institute of Technology, MAHE).*
