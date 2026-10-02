# System Architecture

## Overview
Three core components power the system:
1. **Memory Store** — Central store for all memory units across agents
2. **Decay Engine** — Applies Ebbinghaus forgetting curve to memory scores
3. **Orchestrator** — Routes tasks to agents and manages coordination

## Memory Unit Schema
| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Unique identifier |
| content | str | Memory content |
| agent_id | str | Owning agent |
| tags | list | Semantic labels |
| retention_score | float | Current retention (0–1) |
| access_count | int | Number of retrievals |
| last_accessed | datetime | Last retrieval timestamp |

## Forgetting Curve Formula
R(t) = e^(- λt / S)
- R(t) = Retention at time t
- λ = Decay rate
- t = Time elapsed (hours)
- S = Stability factor (increases with reinforcement)
