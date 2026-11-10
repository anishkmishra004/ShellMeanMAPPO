# ShellMean-MAPPO: Major Project Evaluation Panel Demonstration Script

**Project Title:** ShellMean-MAPPO: A Conflict-Aware Multi-Agent Reinforcement Learning Framework for Downlink Resource Allocation in Multi-Shell LEO Satellite Networks  
**Target Audience:** University Major Project Examination Committee & Review Panel  
**Platform Interface:** OrbitTrack Mission Control (`http://localhost:8501`)  

---

## 🎯 Executive Presentation Outline (10–15 Minutes)

| Phase | Time | Action in Application | Key Talking Points for Panel |
|---|---|---|---|
| **Phase 1: Project Overview & Motivation** | 2 min | Open `📊 Dashboard` | Problem of multi-shell LEO downlink contention, spectrum reuse, and why conventional single-satellite heuristics fail. |
| **Phase 2: Constellation Digital Twin** | 2 min | Show 3D Globe + Orbit-02 Telemetry | Dual-shell Keplerian kinematics (550 km vs 1100 km), line-of-sight elevation masks (25°), and Ka-band 20 GHz RF parameters. |
| **Phase 3: 3-Operation Scenario Generation** | 2 min | Click `🚀 Apply Preset & Run Scenario` | Explain the 3 automated operations: 1) Constellation Sizing, 2) Ground Demand & Queues, 3) Autonomous Downlink Execution. |
| **Phase 4: Conflict Resolution Demonstration** | 2 min | Switch to `⚔️ Conflict & Spectrum` | Dynamic backlog bounds $M_c(t) = \min(M_{c,\max}, \lceil Q_c / Q_{\text{seg}} \rceil)$, multi-attribute scoring $S(i,k,c)$, and deterministic tie-breaking. |
| **Phase 5: Multi-Algorithm Benchmarking** | 3 min | Switch to `🔀 Policy Benchmarking` | Click *Run Controlled Multi-Seed Benchmark*. Compare Rule-Based, Greedy, Random, and ShellMean-MAPPO across identical seeds. |
| **Phase 6: Live MAPPO Training Center** | 2 min | Switch to `🧠 Live MAPPO Training` | Click *Run Live Training Iteration*. Show PPO surrogate loss, value critic MSE, entropy regularization, and GAE advantage estimation. |
| **Phase 7: Scientific Report Export & Conclusion** | 1 min | Switch to `📑 Reports & Export` | Download Markdown report & CSV telemetry. Q&A Defense. |

---

## 🗣️ Verbatim Script & Demonstration Steps

### Phase 1: Problem Definition & Motivation (Dashboard)
> *"Respected committee members, in modern Low Earth Orbit megaconstellations like Starlink and OneWeb, satellites operate across heterogeneous orbital shells with distinct orbital velocities, propagation delays, and footprint sizes.*
>
> *When multiple satellites concurrently target the same terrestrial ground cell using overlapping Resource Blocks (RBs), two critical physical bottlenecks emerge:*
> 1. *Packet Contention: A ground cell cannot accept unlimited simultaneous downlinks without destructive interference.*
> 2. *Co-Channel Interference: Sidelobes leak RF power across shared frequency channels.*
>
> *Our project, **ShellMean-MAPPO**, introduces a deterministic conflict resolver coupled with an autoregressive multi-agent reinforcement learning policy to optimize downlink throughput while ensuring fairness and eliminating co-channel packet collisions."*

---

### Phase 2: Constellation Digital Twin (3D Globe & Subsystems)
*(Point to the central 3D globe and the right telemetry panel)*
> *"Here on the Mission Control console, you observe our physical digital twin:*
> - *We simulate 8 satellites partitioned into Shell 0 (550 km altitude, 53° inclination) and Shell 1 (1100 km altitude, 70° inclination).*
> - *Orbits are propagated via circular Keplerian kinematics accounting for Earth rotation at $7.2921 \times 10^{-5}\text{ rad/s}$.*
> - *On the right panel, you can inspect instantaneous telemetry for Satellite ORBIT-02: mean altitude 542.6 km, velocity 7.61 km/s, solar power status, and our Ka-band 20 GHz downlink locked status with 8 Resource Blocks (200 MHz bandwidth).*"

---

### Phase 3: The 3 Core Operations (Scenario Generation)
*(Select `High Traffic` or `Congested Network` in the sidebar and click **Apply Preset & Run Scenario**)*
> *"When we trigger a scenario preset, the system performs **three fundamental operations** live:*
> 1. *Operation 1 (Kinematic Synthesis): Calibrates orbital parameters, satellite phase offsets, and line-of-sight elevation angles requiring $\theta \ge 25^\circ$.*
> 2. *Operation 2 (Terrestrial Demand Generation): Synthesizes Poisson and bursty traffic arrivals across 20 ground cells, creating dynamic congestion hotspots.*
> 3. *Operation 3 (Autonomous Downlink Execution): Instantly executes scheduling slots, demonstrating beam steering, contention resolution, and buffer queue drainage ($Q(t+1) = \max(0, Q(t) - X(t)) + A(t)$).*"

---

### Phase 4: Deterministic Conflict Resolution
*(Switch to `⚔️ Conflict & Spectrum` in the sidebar)*
> *"One of our primary research contributions is the **deterministic conflict resolution engine**:*
> - *Unlike naive heuristics that accept all requests and cause packet collisions, our resolver enforces a dynamic backlog-dependent admission limit:*
>   $$M_c(t) = \mathbf{1}[Q_c(t) > 0] \cdot \min\left(M_{c,\max}, \left\lceil \frac{Q_c(t)}{Q_{\text{seg}}} \right\rceil\right)$$
> - *Competing satellite requests are ranked deterministically by a multi-attribute score $S(i,k,c)$ combining channel gain, committed transmit power, queue urgency, and shell propagation delay penalty.*
> - *Top $M_c(t)$ requests are retained; conflicting requests are rejected with zero transmission and zero interference.*"

---

### Phase 5: Multi-Algorithm Benchmark Comparison
*(Switch to `🔀 Policy Benchmarking` and click **Run Controlled Multi-Seed Benchmark**)*
> *"To scientifically validate our framework, we evaluate four scheduling algorithms under **strictly identical initial environments, orbital trajectories, and random seeds (42 and 101)**:*
> 1. *Rule-Based Conflict-Aware Policy: Specializes Shell 0 for hotspots and Shell 1 for regular cells with frequency orthogonalization.*
> 2. *Greedy Policy: Selects top channel gain × queue cells, but suffers request contention collisions.*
> 3. *Random Policy: Unstructured baseline boundary.*
> 4. *ShellMean-MAPPO: Decentralized neural network policy with centralized critic training.*
>
> *(Point to the generated table)* *Notice that the Rule-Based policy achieves the highest request retention (98.8%), while our MAPPO policy demonstrates autonomous decentralized coordination without manual rule engineering.*"

---

### Phase 6: Live MAPPO Training Center
*(Switch to `🧠 Live MAPPO Training` and click **Run Live Training Iteration**)*
> *"Here in the training center, we can execute live training iterations:*
> - *The actor network employs an **autoregressive decomposition**: selecting target cell first, then resource block chunk, then continuous power share via a Beta distribution.*
> - *The centralized critic estimates $V(S)$ from a global 194-dimensional network state.*
> - *We resolve multi-agent credit assignment via **Counterfactual Marginal Beam Credit**: replaying each slot with individual beams deactivated ($U(\mathbf{a}) - U(\mathbf{a}_{-i,k})$).*
> - *Notice the live Plotly convergence curves updating with rollout rewards, actor/critic losses, policy entropy, and evaluation throughput.*"

---

### Phase 7: Scientific Export & Summary
*(Switch to `📑 Reports & Export`)*
> *"Finally, all quantitative telemetry can be exported as a publication-ready Markdown technical report and raw CSV data.*
>
> *Our entire software platform has been verified with **37 automated unit and integration tests passing at 100%**, demonstrating a complete, reproducible digital twin for LEO satellite networks.*
>
> *Thank you. We now welcome questions from the examination committee."*

---

## 🛡️ Committee Q&A Defense Guide

### Q1: Why does the Rule-Based heuristic achieve higher throughput than the current MAPPO checkpoint?
**Answer:**  
*"The Rule-Based heuristic was engineered with hardcoded domain knowledge: explicit inter-shell frequency partitioning (Shell 0 strictly on RBs 0–3, Shell 1 strictly on RBs 4–7) and global cell priority sorting. Our MAPPO checkpoint currently represents early-stage training (Epoch 4 / 256 updates). Reinforcement learning discovers coordination through trial and error from a blank state. As detailed in our technical report, full convergence to match or exceed the hand-crafted heuristic typically requires 10,000 to 20,000 training slots. In accordance with research honesty, we report the exact empirical checkpoint performance without synthetic inflation."*

### Q2: How do you model co-channel interference mathematically?
**Answer:**  
*"Interference is evaluated on a per-Resource-Block basis. For an active beam on Satellite $i$ serving cell $c$, interference occurs whenever any other satellite $j$ simultaneously transmits on an overlapping Resource Block. The interference power is:
$$I_{i,c} = \sum_{j \ne i} \mathbf{1}_{\text{overlap}}(RB_i, RB_j) \cdot P_{j} \cdot h_{j \to c}$$
where $h_{j \to c}$ incorporates the off-axis antenna pattern roll-off based on ITU-R parabolic beam standards ($G(\phi) = G_{\max} - 12(\phi / \theta_{3\text{dB}})^2$)."*

### Q3: What is the purpose of Counterfactual Marginal Beam Credit?
**Answer:**  
*"In decentralized cooperative MARL, satellites share a collective team reward $R(t)$. This causes the multi-agent credit assignment problem: a single satellite cannot determine whether a high reward was due to its own smart beam allocation or another satellite's actions. By replaying the slot with a specific beam turned off without mutating the environment state ($U(\mathbf{a}) - U(\mathbf{a}_{-i,k})$), we isolate the exact marginal utility contribution of that beam."*

