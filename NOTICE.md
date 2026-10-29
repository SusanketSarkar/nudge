# Attribution and provenance

Nudge is **inspired by** [Neuron Explorer](https://github.com/lyramakesmusic/neuron-explorer)
(steering-first neuron exploration, random-direction controls, retained rollouts).

**No Neuron Explorer source code is included in this repository.** At the time of writing the
upstream repository ships no LICENSE file, so its code cannot be assumed reusable. Nudge's steering
hook, dose calibration, sampler, controls and scoring are independent implementations written from the
*ideas and mathematics* described in its README and research log (additive steering at the MLP
output along a neuron's down-projection column; doses expressed relative to residual/MLP-output norm;
random-direction controls of matched norm).

If upstream adds a permissive license and you want to reuse parts of it (for example its vLLM-fork
steering path), write a backend in `src/nudge/models/explorer_adapter.py` and keep its attribution here.
