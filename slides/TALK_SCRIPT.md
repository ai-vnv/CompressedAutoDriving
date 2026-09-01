# Talk Script — Internship Summary: From Backend Engineer to Closed-Loop V&V Research

**Occasion**: internship completion, AI Verification & Validation Lab
**Format**: oral, ~17 minutes (fits 15–20), 22 slides
**Slides**: `slides/main.pdf` (Beamer) / `slides/presentation.pptx`

---

## Slide 1: Title [0:00 – 0:15]

*[Wait for introduction]*

"Thank you. I'm Alfan. This talk is the summary of my internship at the AI
V&V Lab — the story of getting from backend engineering to a submitted
verification-and-validation paper, and everything built along the way."

---

## Slide 2: The starting point [0:15 – 1:15]

"Let me be honest about where this started. My background is backend
engineering, not computer-science research. What pulled me in was a question:
how do agents decide under uncertainty — the world of POMDPs.

The first month was rough. The theory, the tooling, the papers — all
unfamiliar ground, and my performance showed it. The useful realization was
that the gap was fundamentals, not effort. So I decided to fix it properly."

→ *Transition*: "Which meant going back to basics, deliberately."

---

## Slide 3: The reset [1:15 – 2:00]

"I restarted from the foundations: a decision-making course on Coursera from
MDP basics up, and the JuliaAcademy course on Decision Making Under
Uncertainty with POMDPs.jl.

And I adopted one rule: every concept gets rebuilt in code before I move on.
Learn, build, verify — that loop became how I worked for the rest of the
internship."

→ *Transition*: "The building part turned into its own artifact."

---

## Slide 4: Learning by building [2:00 – 3:00]

"I wrote hands-on Pluto notebooks covering MDPs and POMDPs from first
principles, with an interactive solver playground to explore the JuliaPOMDP
solver ecosystem side by side, and worked through the Gallery of POMDPs.jl
examples.

The notebooks are public, and they double as a learning medium for whoever
comes to the lab next with the same gap I had."

→ *Transition*: "With the fundamentals in place, the first real project."

---

## Slide 5: First project — DuckieMDP [3:00 – 4:15]

"Project one: formulate Duckietown driving as a fully observed MDP, and build
a certification-oriented explanation framework on top of it. Every explanation
comes in three pillars: why this action, what if another action, and is the
behavior consistent.

The technical core: counterfactual rollouts share the exogenous noise with the
factual episode, so outcome claims are settled exactly rather than estimated.
It works across four policy classes — Q-learning, SARSA, SAC, and TD3 — and
it became an arXiv preprint."

→ *Transition*: "Alongside it grew a side project."

---

## Slide 6: Side project — Duckietown.jl [4:15 – 5:15]

"I reimplemented the environment natively in Julia as a POMDPs.jl problem —
Duckietown.jl, now in the lab's GitHub organization. The part I'm most proud
of: it is validated against the Python original decision by decision,
including the exact NumPy random-number streams, so a seeded episode
reproduces bit for bit across languages. It also has a native renderer, and
the DORASolvers case study drives a full lap in it."

→ *Transition*: "All of that was preparation for the main project."

---

## Slide 7: Main project divider [5:15 – 5:45]

"The main project, now submitted to IEEE Open Journal of Intelligent
Transportation Systems: a closed-loop evaluation of how automated driving
capabilities survive model compression. This is where the POMDP goes from
something I was learning to the instrument of the study."

---

## Slide 8: The problem [5:45 – 6:30]

"Learned driving policies run on embedded computers, so they get compressed —
pruned, distilled, quantized. Progress is measured by size, latency, and
aggregate accuracy. But aggregate scores can hide exactly what matters:
which behaviors compression removes. The real question is which learned
capabilities survive each stage."

---

## Slide 9: Key idea [6:30 – 7:30]

"A driving policy acts on its own observations, so small action drift
compounds through the feedback loop — static test scores cannot see it.

Our answer reverses scenario-based assessment: instead of fixing the vehicle
and varying scenarios, we fix the scenario set and the evaluation seeds, and
vary the system under test. Every compression stage is a new configuration
that must be re-certified against acceptance criteria fixed before any result
existed. That gives us four questions: where does driving break, is it
recoverable, does the data decide the recovery, and does recovery survive
lower precision."

---

## Slide 10: The driving task [7:30 – 8:15]

"Five curricula in Gym-Duckietown, same eight seeds each: lane following on
two loops, a crossing pedestrian, a stop sign — stop, hold, resume — and both
combined. The stop curricula are where everything interesting happens: they
force the policy to interrupt its own driving and re-establish it."

---

## Slide 11: System under test [8:15 – 9:15]

"The policy never sees simulator state. Camera frames go through YOLO11n for
objects and MobileNetV3 for lane pose; extended Kalman filters fuse those into
a 29-dimensional belief vector; and a belief-PPO actor of 73,986 parameters
maps belief to the driving command. Only the actor is compressed — perception
stays fixed, so behavioral changes are attributable to compression."

---

## Slide 12: How the pipeline works [9:15 – 10:15]

"Ten configurations, one ancestry. A1 prunes the hidden width from 256 to 64.
A2 distills on the hardest curriculum only; A3 on balanced data across all
five. From the recovered A3: post-training INT8 (A6), quantization-aware INT8
(A8), and an FP16 cast (A9) as the precision control. A4, A5, A7 isolate
individual factors. Every configuration is evaluated the same way — nothing
inherits trust from the stage before."

---

## Slide 13: Protocol [10:15 – 11:00]

"Four hundred matched episodes — ten models, five curricula, eight identical
seeds. Acceptance checks preregistered. The evaluation backend verified to
reproduce bit for bit. And two independent axes: does it drive, and does it
imitate the original's actions. Keeping them separate is what exposes the
final finding."

---

## Slide 14: Findings 1 & 2 [11:00 – 12:00]

"Finding one: pruning is where driving first breaks — 91.6% of parameters
removed, and all five curricula fail.

Finding two: recovery is decided by the rehearsal data, not by distillation
itself. Same teacher, loss, optimizer, and budget: hardest-curriculum-only
rehearsal recovers just the stop tasks, balanced rehearsal recovers
everything. What the student rehearses is what the student keeps."

---

## Slide 15: Finding 3 [12:00 – 13:00]

"This matrix is the study in one picture. Top to bottom: pruning fails
everything, partial rehearsal recovers a corner, balanced rehearsal recovers
the full row — and then both INT8 routes, from one byte-identical recovered
checkpoint, fail exactly the two stop curricula again. Stop-task completion
falls from eight of eight to three of eight under PTQ."

---

## Slide 16: Opposite phenotypes [13:00 – 14:00]

"And they fail in opposite ways. The PTQ actor brakes correctly, parks 22
centimeters before the line, and never moves again — all the way to the
horizon. The frames here are reconstructions placed from the recorded
telemetry; the circles on the curves are those same steps. The QAT actor does
the opposite: it drives through the stop at low speed. One forgets how to go,
the other forgets how to stay."

---

## Slide 17: Findings 4 & 5 [14:00 – 15:00]

"The controls isolate the cause. The same PTQ on the unpruned actor passes
everything; FP16 on the recovered actor passes everything. So the failure is
the interaction of narrow width with integer quantization — under the tested
procedures — not precision alone.

And the dissociation: QAT imitates the original more closely than PTQ yet
drives worse, while the unpruned INT8 control drives everything yet fails the
similarity thresholds. Action similarity is not an acceptance test, in either
direction."

---

## Slide 18: Cost [15:00 – 15:45]

"What compression buys here: FP16 halves parameter memory but runs 26% slower
on this CPU — a memory move, not a speed move. INT8 is fastest and smallest
but fails the stop curricula. And the actor is under 0.1% of end-to-end step
cost — perception dominates, which points directly at the future work."

---

## Slide 19: Takeaways [15:45 – 16:30]

"The takeaways: compression of an acting policy is a sequence of behavioral
transitions — lose, recover, lose again. Evaluate every stage in closed loop.
Choose rehearsal data to cover everything the policy must keep. Don't assume a
completed recovery survives a precision cut. And don't use action similarity
as an acceptance test. In one sentence: judge compressed policies by how they
drive."

---

## Slide 20: Future works [16:30 – 17:15]

"Next steps, in order of ambition: widen the settings — more quantization
configurations, training realizations, and seeds. Compress the perception
stack, the actual 99.9% of step cost, under the same acceptance protocol.
Transfer the protocol to other stacks and simulators, eventually hardware.
And the Julia-native line: Duckietown.jl gives the lab a ground for
solver-level V&V experiments with bit-for-bit reproducibility built in."

---

## Slide 21: What the internship built [17:15 – 17:45]

"So, what did the internship build? A path from backend engineering to a
working RL and POMDP research workflow. Four public repositories, two papers,
one Julia package. From a bad first month to a journal submission. And the
lesson that stuck: measure what matters, in closed loop."

---

## Slide 22: Thank you [17:45]

"Everything in this talk is public — the four repositories are on screen.
Thank you, and I'm happy to take questions."

---

## Time budget

| Slide | Topic | Duration | Cumulative |
|:--:|--|:--:|:--:|
| 1 | Title | 0:15 | 0:15 |
| 2 | Starting point | 1:00 | 1:15 |
| 3 | The reset | 0:45 | 2:00 |
| 4 | Learning by building | 1:00 | 3:00 |
| 5 | DuckieMDP | 1:15 | 4:15 |
| 6 | Duckietown.jl | 1:00 | 5:15 |
| 7 | Divider | 0:30 | 5:45 |
| 8 | Problem | 0:45 | 6:30 |
| 9 | Key idea | 1:00 | 7:30 |
| 10 | Task | 0:45 | 8:15 |
| 11 | System under test | 1:00 | 9:15 |
| 12 | Pipeline | 1:00 | 10:15 |
| 13 | Protocol | 0:45 | 11:00 |
| 14 | Findings 1 & 2 | 1:00 | 12:00 |
| 15 | Finding 3 | 1:00 | 13:00 |
| 16 | Phenotypes | 1:00 | 14:00 |
| 17 | Findings 4 & 5 | 1:00 | 15:00 |
| 18 | Cost | 0:45 | 15:45 |
| 19 | Takeaways | 0:45 | 16:30 |
| 20 | Future works | 0:45 | 17:15 |
| 21 | Reflection | 0:30 | 17:45 |
| 22 | Thanks | 0:15 | 18:00 |

**Total**: ~18:00 (target 15–20). To hit 15:00, compress slides 10–13 to ~30 s
each and trim slide 4.

---

## Anticipated Q&A

### Q1: What was the hardest part of the transition from backend engineering?
**A**: "Reading research honestly — knowing when I understood a concept versus
when I only recognized the words. The fix was the rebuild-it-in-code rule:
if I couldn't implement it in a notebook, I didn't understand it yet."

### Q2: How do the two papers relate?
**A**: "Same driving world, opposite observability. DuckieMDP is fully
observed, which is what makes exact counterfactual certification possible.
The main paper goes partially observed — belief-state POMDP — and asks an
evaluation question instead of an explanation question. The first project
taught me the formulation discipline the second one needed."

### Q3: Why reimplement the simulator in Julia?
**A**: "Two reasons: the lab's solver ecosystem is Julia-native (POMDPs.jl),
and V&V needs reproducibility — the port is validated decision by decision
against Python including exact RNG streams, so a seeded episode reproduces
bit for bit. That makes it a trustworthy ground for solver-level experiments."

### Q4: Is the INT8 failure a property of quantization in general?
**A**: "No — the controls show that. The same PTQ on the unpruned actor passes
everything, and FP16 on the recovered actor passes everything. The failure is
the interaction of the narrow recovered actor with integer quantization,
under the tested procedures."

### Q5: Eight seeds — what can you resolve?
**A**: "Large effects, like 8/8 versus 3/8, which is what we report. The paper
carries an explicit statistical-resolution paragraph, and a reserved
evaluation split was never opened because no candidate met the full
deployment criterion."

### Q6: Why compress the actor if it's under 0.1% of step cost?
**A**: "Because it isolates the question — with perception fixed, every
behavioral change is attributable to compression. The evaluation protocol is
the contribution, and it applies unchanged to compressing perception, which
is the expensive part and the stated future work."

### Q7: Why is FP16 slower than FP32?
**A**: "No native half-precision path on this CPU, so weights are converted
during inference — 26% more latency for half the memory. We present FP16
strictly as a memory option."

### Q8: What would you do differently if you started the internship again?
**A**: "Start with the fundamentals reset in week one instead of month two —
and start writing evaluation code with preregistered acceptance checks from
the first experiment, because that discipline is what made the main project's
results defensible."
