# New episode audit — 2 October 2026

Raw input: `/media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/staging/episodes`.
Baseline: `ur5_cube_clean_73ep_v2/meta/ur5_vla_conversion.json`.

Found **39 new finalized episodes / 92,290 frames**. There are 112 finalized episodes in total, including all original 73. No incomplete folders were found at inspection.

**Selected: 37 new episodes / 88,497 frames.** All selected episodes passed the existing raw validator with full media decoding. Every RGB and depth reference in all 39 new episodes was decoded: 369,160 media references. Check full_media_report.json for per-episode results. Additional checks cover a single open–close–open sequence, starting configuration, task consistency, backwards image timestamps, and prolonged unchanged image timestamps.

## Exclusions from the proposed selection

- `episode_20261001_175654` (3,195 frames): 1 frame interval(s) exceed 0.20s; largest is 0.200162s between frames 231 and 232.
- `episode_20261001_182413` (598 frames): no gripper close transition; no gripper open transition; not a single open-close-open sequence.

The 0.200162 s gap is only 0.000162 s above the existing 0.20 s threshold. It is a strict validator failure, not evidence of a multi-second pause. The other excluded episode is about 30 s long, has no gripper transition, and its sampled images show an incomplete approach. Both original episode directories are preserved.

## Other observations

- Maximum consecutive arm-target change over all new episodes: 0.0042744 rad.
- Maximum commanded-versus-measured joint difference: 0.0060728 rad.
- Maximum initial deviation from standard pose: 0.0063539 rad.
- Accepted episodes contain 289 dropped samples reported by metadata; remaining frame intervals stay within the existing 0.20 s limit. Dropped samples should still be reported in the thesis.
- The largest per-episode 99th-percentile base/wrist source timestamp skew is approximately 0.0729 s. Receive-age validation is not exact camera synchronization.
- Four sampled times per episode, each with both views, were visually reviewed (start, close, open, final). Completed attempts appear to end with the orange cube in the bowl. The failed short episode does not. The new recordings contain blue/green distractor cubes and visibly varied cube positions.
- Distinct base and wrist views are present in the reviewed snapshots. No image/depth decoding errors were reported by the completed validator.
- This is full file decoding plus sampled visual review, not frame-by-frame human annotation or independent physical success measurement.

## Conversion selections

`accepted_new_episodes.txt` selects only the 37 accepted new recordings.
`combined_episodes.txt` preserves the original 73 and adds those 37: 110 episodes / 285,204 frames.

Historical advisory: the original `episode_20260829_200428` has four gripper transitions. It remains in the combined list to preserve the previous 73-episode selection. To require exactly one grasp/release cycle for every episode, exclude it explicitly; the resulting selection is 109 episodes / 282,588 frames. The original 73 were not fully revalidated in this audit. Conversion preflight fully validates all selected input episodes.

The commands use a new destination and do not overwrite the existing converted dataset. Conversion itself has not been run. The local `pi0` environment was checked: LeRobot 0.5.2 imports, and the installed dataset creation, save_episode and finalize interfaces are compatible with the converter. The converter currently writes a historical 0.4.3 target label into its metadata; the actual installed version checked here is 0.5.2.
