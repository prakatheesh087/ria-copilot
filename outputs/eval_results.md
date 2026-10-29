# Evaluation results (measured)

Environment: Python 3.13.17 · Linux-6.18.44-fc-v185-x86_64-with-glibc2.39 · offline (no API key): templates + extractive document answers · app 1.1.0 · rule set 2026.10.29

| Area | Metric | Result |
|---|---|---|
| Financial maths | Closed-form checks passed | 4/4 |
| Compliance (held-out) | Unsafe requests blocked | 15/15 (100%) |
| Compliance (held-out) | Legitimate questions wrongly refused | 0/15 (0%) |
| Compliance (brief) | Persona probes correct | 13/13 |
| Document Q&A | Top-1 retrieved page correct | 100% of 8 |
| Document Q&A | First citation on the correct page | 100% of 8 |
| Document Q&A | Unanswerable questions → insufficient evidence | 3/3 |
| Latency | Plan generation, median / p95 | 236 / 355 ms |
| Latency | Document answer (offline), median / p95 | 0.2 / 0.4 ms |

Missed unsafe requests: none

Wrongly refused: none

Limits: small hand-written sets; synthetic 5-page SID; offline mode only. These numbers show the rules behave as designed on these cases; they are not evidence of regulatory compliance.