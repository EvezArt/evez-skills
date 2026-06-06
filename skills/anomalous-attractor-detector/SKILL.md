---
name: anomalous-attractor-detector
description: "Uses strange attractors, self-organized criticality, and anomalistic psychology metrics to flag UAP-like phase transitions in behavioral game theory data."
version: "1.0.0"
category: AI
author: EVEZ-OS / Steven Vearl Crawford-Maggard
---

# AnomalousAttractorDetector Skill

## Overview
Detects strange attractors and phase transitions in scalar time-series data.
Flags UAP-like signatures: power-law departures from SOC band + chaotic Lyapunov exponent.

## Use When
- Anomaly detection in time-series (atmospheric, behavioral, financial)
- UAP signature research (phase transition detection)
- Self-organized criticality verification
- Detecting runaway divergence before it becomes catastrophic

## Theory
At self-organized criticality (SOC), event sizes follow power laws with α ∈ [1.5, 2.5].
Departures signal supercritical runaway (α < 1.5) or subcritical collapse (α > 2.5).
Positive Lyapunov exponent confirms chaotic (strange) attractor.

## Anomaly Flag Criteria (ALL must be true)
1. Lyapunov λ > 0.05 (chaotic regime)
2. Power-law α outside SOC band [1.5, 2.5]
3. Phase transition risk > 0.8 (variance divergence)

## Implementation
```python
from src.rqns.attractor import AnomalousAttractorDetector
detector = AnomalousAttractorDetector()
scan = detector.scan(time_series_array)
if scan.anomaly_flag:
    print(f"UAP-signature detected: α={scan.power_law_alpha:.2f}, λ={scan.lyapunov_estimate:.4f}")
```

## Output
- `lyapunov_estimate` — > 0 = chaotic
- `power_law_alpha` — SOC exponent (target: 1.5–2.5)
- `soc_score` — [0,1] proximity to criticality
- `phase_transition_risk` — [0,1] variance divergence
- `anomaly_flag` — True if UAP-signature detected

