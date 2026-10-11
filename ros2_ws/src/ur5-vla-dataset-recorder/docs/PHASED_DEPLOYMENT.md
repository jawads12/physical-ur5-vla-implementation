# Phased Policy Deployment and Safety Checklist

## Phases

1. **A — offline:** Run the checkpoint on saved observations, inspect ranges/jumps, and compare demonstrated/predicted CSV data.
2. **B — shadow:** Read real cameras, named joint state, estimated gripper state, Servo status, controller/readiness state, and deadman; log predictions only. Create no command publishers.
3. **C — simulation/mock:** Exercise disable, stale-data, timeout, controller mismatch, communication loss, NaN, limits, velocity, workspace, deadman, and shutdown faults. Verify every fault clears chunks and latches.
4. **D — physical/no objects:** After code and risk review, use low speed and one bounded action. Stop and inspect; never execute a whole chunk.
5. **E — guarded in-distribution:** One attempt with the trained object, bowl, cameras, lighting, instruction, and start pose.
6. **F — evaluation:** Repeated, separately recorded trials with predefined success and safety metrics.

The runnable two-process procedure for Phase B is documented in [LIVE_SHADOW_INFERENCE.md](LIVE_SHADOW_INFERENCE.md).

Advancing a phase requires explicit human approval. A successful conversion or training run never authorizes robot motion.

## Clean retraining dataset and checkpoint gates

The retraining source is `local/ur5_cube_clean_20ep`: 20 episodes and 32,022 frames. Its immutable selection manifest is `config/ur5_cube_clean_20ep.json`. The original raw archive is retained; no source episode is deleted.

Training saves complete checkpoints at 5,000-step intervals through 40,000. Do not assume the 40,000-step checkpoint is best. Evaluate `005000`, `010000`, ..., `040000` and select the lowest-error checkpoint that passes every gate below:

1. **Artifact gate:** model, processor, config, normalization statistics, and training-step metadata exist and hash correctly.
2. **Offline data gate:** predictions are finite on samples from every episode; camera/task/state schema is exact; compare predicted first action with the recorded action and current state. A candidate fails if any arm target delta exceeds the demonstrated dataset maximum of 0.012566 rad.
3. **Shadow gate:** run the real cameras and robot state without publishers. Require at least 100 consecutive valid predictions, no stale inputs or Servo faults, every target delta at or below 0.012566 rad, and prediction-to-prediction jitter at or below 0.005 rad.
4. **Guarded motion gate:** only after explicit human approval, begin with no object, UR speed slider low, emergency stop reachable, a continuously held physical deadman, one checked target, and an immediate stop/inspection. Never execute a complete predicted chunk.

Clipping or discarding an unsafe prediction is a last-line safety response, not evidence that the policy is valid. Count and report every rejection. Any rejection during the initial 100-prediction shadow run blocks physical testing.

## Live adapter acceptance criteria

- Continuously held physical deadman; release stops publication immediately.
- Starts disabled and requires deliberate manual enable.
- Controller is `forward_position_controller`; External Control and Servo readiness are positively confirmed.
- First six policy values are absolute joint targets and the seventh is an absolute binary gripper request.
- Named-joint reordering matches training exactly.
- The same checkpoint preprocessing and postprocessing are used.
- Closed-loop/receding-horizon inference executes one checked action, discards the rest, and observes again.
- Separate supervisor checks finite 7D output, target delta, configured joint limits, velocity, sensor/inference freshness, Servo zero, controller/readiness, workspace, timeout, deadman, and communication health.
- The adapter defines and tests its controller-appropriate zero-motion/hold behavior. It never switches controllers or sends unreviewed commands directly to the UR motor controller.
- Every failure stops publication, clears queued actions, latches a fault, and requires inspection plus manual re-enable. There is no automatic resume.

## First physical test

- Same cameras, serials, calibration, lighting, orange cube, bowl, start pose, and language instruction as training.
- Clear workspace and verified configured joint/workspace limits.
- Low robot speed; emergency stop reachable; one observer solely responsible for safety.
- Human continuously holds the physical deadman.
- Verify expected controller, External Control, Servo zero, fresh sensors, and policy timeout behavior before enabling.
- Execute one bounded action, stop, disable, and inspect.
- Only after review, allow a few individual closed-loop actions, stopping between them.
- Record the complete evaluation episode and all safety events.

## Emergency stop procedure

1. Release the deadman.
2. Press the UR emergency stop if motion does not immediately cease or any hazard exists.
3. Do not re-enable automatically. Clear the workspace and identify the latched software/hardware fault.
4. Verify controller, External Control, Servo, communications, sensor freshness, and queued-action state.
5. A human may clear the fault only after inspection; repeat the no-object single-action phase after any safety-related change.

The pneumatic gripper state is only the last successfully acknowledged SetIO command. It cannot establish finger position, contact, force, object retention, or successful release.
