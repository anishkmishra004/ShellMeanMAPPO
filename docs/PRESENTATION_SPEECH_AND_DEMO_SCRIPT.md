# ShellMean-MAPPO: Final Major Project Presentation Speech & Live Demo Script
**Project Title**: ShellMean-MAPPO: Conflict-Aware MARL for Multi-Shell LEO Downlink Scheduling  
**Project Guide**: Dr. D. Baswaraj  

**Team Members & Roles**:
1. **Mohammad Azharuddin** (`1602-23-733-310`) — **Person 1** *(Slides 1 to 5: Introduction, Problem Statement, Literature & Physical Channel Model)* — ~3 Mins
2. **A. Shivaleela** (`1602-23-733-314`) — **Person 2** *(Slides 6 to 10: Proposed Architecture, Critic Aggregation, Masking, Conflict Resolution & Reward)* — ~3 Mins
3. **Anish Kumar Mishra** (`1602-23-733-309`) — **Person 3** *(Slides 11 to 15: Implementation Algorithm, Results, Ablations & Conclusion + Complete Live Dashboard Walkthrough)* — ~5 Mins

---

## 🎙️ PERSON 1: Mohammad Azharuddin (1602-23-733-310)
### Coverage: Slides 1 to 5 (Title, Introduction, Problem Statement, Literature & System Model)
**Target Time**: ~3 Minutes

*(Stand confidently, make eye contact with Dr. Baswaraj and the evaluation panel, and speak in a clear, natural conversational tone.)*

"Good morning respected external examiner, internal panel members, and our project guide Dr. D. Baswaraj sir. 

I am **Mohammad Azharuddin**, and together with my project partners **A. Shivaleela** and **Anish Kumar Mishra**, we are presenting our major project: **ShellMean-MAPPO: Conflict-Aware Multi-Agent Reinforcement Learning for Multi-Shell LEO Downlink Scheduling**.

Now, starting with **Slide 2**... to give you an intuitive understanding of the domain we are addressing. When we look at next-generation 6G non-terrestrial networks—and commercial constellations like Starlink or Kuiper—they don't just fly satellites at one single height anymore. They deploy what we call **Multi-Shell constellations**. In our project, we model two distinct shells: Shell 0 at 550 kilometers altitude, and Shell 1 at 1100 kilometers. 

Now, these satellites fly at extreme orbital speeds—around 7.6 kilometers per second. That means a satellite stays in the line of sight of a ground station for just about 5 to 10 minutes. Within this tiny time window, each satellite has to make three coupled decisions every single second: *which ground cell to serve, which frequency resource blocks to allocate, and how much RF transmit power to inject*. Because the satellites are moving so rapidly, traditional centralized scheduling from a ground station simply fails—the signal propagation delay is too high. So, the intelligence has to live **onboard the satellite** itself.

Moving to **Slide 3**, this brings us to our core **Problem Statement**.  
Basically, when multiple satellites across these different altitude shells fly over the same geographic area, four severe challenges occur:
- First, **inter-shell beam conflicts and co-channel interference**. If a satellite in Shell 0 and another in Shell 1 both point their directional beams at the same city using the same frequency blocks, they essentially jam each other. The interference spikes, and data packets get destroyed.
- Second, **high-dimensional action coupling**. Trying to optimize cell choice, bandwidth blocks, and continuous power all at once creates an explosive action space.
- Third, **non-stationarity**. In multi-agent systems, when multiple satellites update their neural policies simultaneously, the environment keeps shifting, causing standard RL to destabilize.
- And fourth, **traffic asymmetry**. Real-world ground demand is never uniform—some cities are heavy traffic hotspots, while others are light. Uncoordinated satellites greedily chase the same hotspot and completely starve other ground cells.

On **Slide 4**, when we analyzed existing literature, classical optimization methods like Lagrangian relaxation or fractional programming are far too computationally heavy to compute in real time on a satellite flight computer. On the other hand, greedy heuristics or standard Independent PPO cause severe beam collisions on hotspot cells because satellites act selfishly. What was missing in existing research is a **deterministic onboard conflict resolution layer** and a scalable multi-shell critic.

Finally, on **Slide 5**, here is our **Mathematical System Model**. We model the satellite positions in true 3D Earth-Centered Earth-Fixed (ECEF) coordinates with a minimum elevation angle of 25 degrees. We model a physical Ka-band 20 Gigahertz downlink with 200 Megahertz bandwidth sliced into 8 Resource Blocks. As you can see from the equations on the slide, we calculate the exact Free Space Path Loss plus ITU atmospheric attenuation, calculate co-channel interference from all other transmitting beams, and use Shannon capacity to compute the exact achievable megabits delivered in every simulation slot.

Now, to explain how our proposed **ShellMean-MAPPO** architecture and conflict resolver solve these challenges, I will hand over the presentation to **A. Shivaleela**."

---

## 🎙️ PERSON 2: A. Shivaleela (1602-23-733-314)
### Coverage: Slides 6 to 10 (Proposed Architecture, Critic Aggregation, Action Masking, Conflict Engine & Rewards)
**Target Time**: ~3 Minutes

*(Acknowledge Azharuddin, smile at the panel, and transition smoothly.)*

"Thank you, Azhar. Good morning respected examiners, I am **A. Shivaleela**.

Picking up from **Slide 6**, let me walk you through the proposed methodology: **ShellMean-MAPPO**.  
Our system is built on the **Centralized Training with Decentralized Execution (CTDE)** paradigm. What this means in plain words is: during actual satellite flight, each satellite executes its decisions 100% locally using its onboard actor network. There is zero inter-satellite communication overhead during inference. However, during ground simulation training, a centralized critic evaluates the global constellation to teach the satellites how to coordinate effectively.

As shown in the block diagram on Slide 6, our system has three key tiers:
1. The **Onboard Autoregressive Actor**,
2. The **Deterministic Conflict Resolution Engine**, and
3. The **Shell-Mean Centralized Critic**.

Now, coming to **Slide 7**, here is one of our primary innovations: **Shell-Mean State Aggregation**.  
Normally, if you build a centralized critic for a constellation, concatenating all satellite states makes the input vector scale linearly—$O(|S|)$. If you add more satellites, the neural network architecture breaks.  
We solved this by leveraging the structural symmetry within each orbital shell. As shown in the formula, we take the latent embeddings of satellites inside Shell 0 and average them into a single shell vector $\mathbf{h}_{shell\_0}$, and do the same for Shell 1. This creates a **permutation-invariant** global state. Whether our constellation has 12 satellites or 50 satellites, the critic dimension remains completely constant, and Generalized Advantage Estimation converges much faster.

Moving to **Slide 8**, let's look at how the satellite actually picks actions onboard.  
Instead of outputting cell, frequency, and power all in one giant step, we use **Autoregressive Action Decoupling**:
First, the satellite selects the target cell. Conditioned on that cell, it picks the contiguous resource block span. And conditioned on both, it determines the continuous transmit power.  
To ensure satellites never make illegal moves, we apply **Dynamic Feasibility Action Masking**. If a ground cell is below the 25-degree elevation horizon—meaning it is physically invisible—or if its data queue is empty, the mask sets its probability to negative infinity. This mathematically guarantees that 100% of the sampled actions are physically valid.

Now, on **Slide 9**, let's discuss our **Conflict Resolution Engine**.  
Even with smart reinforcement learning, two satellites might occasionally target the same ground station on overlapping frequencies. To completely eliminate signal destruction, our deterministic conflict resolver intercepts beam requests before RF transmission. Each request receives an admission score based on queue backlog, estimated SINR, and distance. Furthermore, we use a dynamic backlog threshold $\Gamma(\bar{Q})$—when ground queues are heavily congested, admission tightens to prioritize high-backlog cells, and conflicting lower-priority beams are cleanly dropped.

Finally, on **Slide 10**, how do we reward the agents?  
If you give a single team reward, you get what is known as the 'lazy agent problem', where weak satellites rely on strong ones. If you give purely selfish local rewards, satellites blast maximum power and cause heavy interference for neighboring shells.  
We solved this with a **Counterfactual Marginal Beam Credit Reward**. The reward given to satellite $s$ is strictly the total network throughput with its beam, minus what the network throughput would have been without its beam, minus a power penalty. We also incorporate Jain's Fairness Index. This mathematically penalizes any satellite that causes co-channel interference and rewards satellites that drain congested queues.

Now, to present our algorithmic implementation, experimental results, and demonstrate our live interactive digital twin, I will hand over to **Anish Kumar Mishra**."

---

## 🎙️ PERSON 3: Anish Kumar Mishra (1602-23-733-309)
### Coverage: Slides 11 to 15 (Algorithm, Benchmark Results, Ablations & Conclusion) + Complete Live Dashboard Walkthrough
**Target Time**: ~5 Minutes (2 mins on PPT + 3 mins on Live Dashboard)

*(Take over with energy and confidence, address the panel clearly, and transition from the slides to the live browser demo.)*

### Part A: Presentation Slides (Slides 11 – 15)

"Thank you, Shivaleela. Good morning panel members, I am **Anish Kumar Mishra**.

Looking at **Slide 11 and 12**, here you see our complete algorithmic execution and experimental parameters. We implemented the environment in PyTorch and Gymnasium, modeling 12 LEO satellites across two orbital shells at 53 degrees and 98.7 degrees inclination. We simulated a Ka-band 20 Gigahertz downlink with 200 Megahertz bandwidth divided into 8 Resource Blocks, communicating with 10 geographic ground stations subjected to non-homogeneous Poisson burst traffic. All algorithms were evaluated across multiple fixed seeds—42, 101, and 2026—for scientific repeatability.

Now, please direct your attention to the benchmark comparison table on **Slide 13**, which represents the core results of our research:
- **Total Delivered Traffic**: ShellMean-MAPPO achieved **486.4 Megabits**, which is **+42.2% higher** than the rule-based heuristic and more than double the Greedy policy.
- **Why does Greedy perform poorly?** Because Greedy satellites all greedily target the same high-queue hotspot cells, generating catastrophic co-channel interference and dropping SINR down to 11.2 dB.
- **Conflict Retention Rate**: Our method achieved **94.8% collision-free retention**, proving that our conflict resolver and MAPPO policy learn coordinated spatial separation.
- **Fairness**: We achieved a **Jain's Fairness Index of 0.91**, and reduced total terminal backlog by **89%**.

On **Slide 14**, our ablation studies validate every single design choice:
- Disabling the conflict resolution layer drops throughput by **36.4%** due to unmitigated collisions.
- Removing counterfactual marginal credit increases network interference by **4.2 times**.
- And our Shell-Mean critic converges in just **45 epochs**, whereas standard MAPPO takes 110 epochs and independent PPO diverges completely.

To conclude on **Slide 15**, ShellMean-MAPPO proves that conflict-aware MARL with autoregressive masking and shell-mean pooling can reliably solve real-time downlink scheduling for next-generation multi-shell satellite constellations."

---

### Part B: Live Dashboard Demonstration Walkthrough
*(Anish smoothly switches screen to the browser running `http://localhost:8501`)*

"Now, Dr. Baswaraj sir and respected panel members, instead of just showing you static graphs, we built a complete, interactive, research-grade Digital Twin called **OrbitTrack Mission Control**. Let me give you a quick walkthrough of the running platform:

#### 1. Top Header Bar & Status
*'Right here at the top is our mission control status bar. You can see the system is running live, status is ONLINE, with real-time UTC timestamp and telemetry feeds.'*

#### 2. Selected Satellite Banner
*'Directly beneath the header is the active satellite banner. Right now, it is tracking **ORBIT-02** in Shell 0 at 542.6 kilometers altitude, moving at 7.61 kilometers per second. It displays live NORAD ephemeris, next pass timing, and the exact sub-satellite geographic coordinates—currently over Latitude 12.6 degrees North, Longitude 45.2 degrees East.'*

#### 3. Photorealistic 3D Earth Digital Twin (Center Screen)
*'Now, looking at the main center display, this is our **Photorealistic 3D Earth Digital Twin**, rendered using hardware-accelerated WebGL with real NASA 2K Blue Marble textures, specular ocean reflections, and rotating atmospheric clouds.*  
*If you look closely at the 3D instances rendered here:*
- *You can clearly see the **3D orbital rings** circling the globe—cyan for Shell 0 at 550 kilometers, and amber for Shell 1 at 1100 kilometers.*
- *Every satellite in our constellation is modeled in 3D space with its deployed photovoltaic solar wings and status beacons.*
- *On the surface of the Earth, you can see our **Ground Station instances**—emerald green rings for standard terminals and red rings for high-traffic hotspots.*
- *Connecting the satellite to the ground cell, you see **active 3D laser downlink beams** with animated photon packets streaming down in real time.*
- *And as an operator, I can click and drag with my mouse to smoothly rotate the Earth in full 360 degrees, or use the scroll wheel to zoom right in to inspect ground stations up close.'*

#### 4. Subsystem Telemetry Panel (Right Side)
*'On the right panel, we have four live engineering tabs:*
- *The **Telemetry tab** shows mean altitude, period, and visible ground cells in line-of-sight.*
- *The **Attitude tab** shows roll, pitch, and yaw angles.*
- *The **Power tab** shows 98% solar array output and battery voltage.*
- *And the **Comm tab** confirms Ka-band 20 GHz carrier lock, mean downlink SINR, and spot beam transmission status.'*

#### 5. Live Simulation Control (Sidebar)
*'Over in the left sidebar, we can select our policy—such as **ShellMean-MAPPO**—or apply pre-configured scenario presets like **High Traffic** or **High Interference**.*  
*Now, watch what happens when I click the **Step 1** button...*  
*(Click 'Step 1' on the UI)*  
*The simulation advances by one orbital slot. The satellites advance along their true anomalies, new Poisson traffic arrives at ground cells, the neural policy allocates beams, the conflict resolver verifies zero collisions, and our telemetry updates live.'*

#### 6. Bottom Deck: Passes, Live Imagery & Mission Status
*'Looking at the bottom three cards:*
- *On the left: **Upcoming Passes & Ephemeris**, calculating AOS, LOS, and elevation angles for the next 24 hours.*
- *In the center: our **Live Imagery & Target Terrain** card. When I click the **Live Imagery** tab, it displays the actual high-resolution Earth terrain captured by the satellite's onboard optical sensor over ground cell C02 with 1.2-meter resolution and 12.8-kilometer swath width. We can also view **Recent Images** over other ground cells or check the **Downlink Spectrum** table.*
- *On the right: **Mission Status & Timeline**, showing payload sensor states and the live simulation event log.'*

#### 7. Multi-Algorithm Benchmarking & Report Export
*'Finally, in the sidebar navigation, if we click **Policy Benchmarking**, we can run a live multi-seed empirical comparison between ShellMean-MAPPO, Greedy, and Rule-Based policies. And under **Reports & Export**, the platform compiles the complete scientific experiment report and telemetry logs, ready for export as Markdown or CSV.*

*This concludes our presentation and live demonstration. Thank you Dr. Baswaraj sir and respected panel members for your time. We are now ready to take your questions.'*
