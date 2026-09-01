# F10-PPO V25 — Retained C4 student with privileged guidance

V24 is retained as a training-only negative result. Its warm start passed C2,
but global actor changes caused C3 restart failure and a C4 collision. No V24
development or final seed was used.

V25 initializes the warm-start student from the frozen V22 C4 checkpoint,
which already passed C4 development, rather than forcing the V20 C3 actor to
move globally toward C4. V22 is itself a descendant of the selected V20 C3
checkpoint. The V24 public-belief dataset then rehearses C2/C3, C4 teacher and
DAgger states, and exactly one privileged-guided C4 episode. The actor uses a
smaller 1e-4 learning rate for eight epochs. The critic is supervised for 20
epochs at 5e-4 on the guided episode's discounted returns.

The student actor and critic still receive exactly the fixed public 29D belief.
Simulator truth exists only in the offline teacher's source CSV; it is absent
from the NPZ and from deployment. Step-zero C2/C3/C4 training-only gates must
pass before canonical on-policy PPO. PPO and its selection thresholds are
unchanged from V24, and step zero remains ineligible.

V25 uses training seeds 173001..173012, development 173101..173104, and
stage-final 173201..173204. Stop after a once-only C4 stage-final result.
