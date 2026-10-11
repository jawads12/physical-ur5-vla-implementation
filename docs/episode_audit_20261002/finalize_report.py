from pathlib import Path
import csv,json
P=Path(__file__).resolve().parent
PROJECT=P.parents[1]
struct=json.loads((P/'structural_report.json').read_text())
full={r['episode']:r for r in json.loads((P/'full_media_report.json').read_text())}
assert len(struct)==39 and set(full)=={r['episode'] for r in struct},'Full media checks are not yet complete'
accepted=[r for r in struct if r['structural_pass'] and not r['review_notes'] and full[r['episode']]['full_media_pass']]
rejected=[r for r in struct if r not in accepted]
old=json.loads((PROJECT/'data/ur5_vla_dataset/lerobot/ur5_cube_clean_73ep_v2/meta/ur5_vla_conversion.json').read_text())['source_episode_names']
(P/'accepted_new_episodes.txt').write_text('\n'.join(r['episode'] for r in accepted)+'\n')
(P/'combined_episodes.txt').write_text('\n'.join(old+[r['episode'] for r in accepted])+'\n')
frames=sum(r['frames'] for r in accepted)
summary={'new_episodes':39,'new_frames':sum(r['frames'] for r in struct),'selected_new_episodes':len(accepted),'selected_new_frames':frames,'full_media_files_checked':4*sum(r['frames'] for r in struct),'combined_with_original_73_episodes':len(old)+len(accepted),'combined_frames':196707+frames,'excluded_new':{r['episode']:r['errors']+r['review_notes']+list(set(full[r['episode']]['errors'])-set(r['errors'])) for r in rejected},'historical_advisory':'Original episode_20260829_200428 has four gripper transitions. combined_episodes.txt preserves all original 73; a stricter single-attempt variant can exclude it explicitly. Original episodes were not fully revalidated in this audit; converter preflight revalidates any selected originals.'}
(P/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
cols=['episode','frames','duration_s','dropped_frames','structural_pass','full_media_pass','selected','max_gap_s','max_action_step_rad','max_state_action_error_rad','start_pose_error_rad','camera_skew_p99_s','gripper_transitions','issues']
with (P/'episode_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
 for r in struct:
  row={k:r[k] for k in cols if k in r};row.update(full_media_pass=full[r['episode']]['full_media_pass'],selected=r in accepted,gripper_transitions=len(r['transitions']),issues='; '.join(r['errors']+r['review_notes']))
  w.writerow(row)
text=f'''# New episode audit — 2 October 2026

Raw input: `{PROJECT/'data/ur5_vla_dataset/staging/episodes'}`.
Baseline: `ur5_cube_clean_73ep_v2/meta/ur5_vla_conversion.json`.

Found **39 new finalized episodes / {summary['new_frames']:,} frames**. There are 112 finalized episodes in total, including all original 73. No incomplete folders were found at inspection.

**Selected: {len(accepted)} new episodes / {frames:,} frames.** All selected episodes passed the existing raw validator with full media decoding. Every RGB and depth reference in all 39 new episodes was decoded: {summary['full_media_files_checked']:,} media references. Check full_media_report.json for per-episode results. Additional checks cover a single open–close–open sequence, starting configuration, task consistency, backwards image timestamps, and prolonged unchanged image timestamps.

## Exclusions from the proposed selection

'''
for r in rejected:text+=f"- `{r['episode']}` ({r['frames']:,} frames): "+'; '.join(r['errors']+r['review_notes'])+'.\n'
text+='''
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
'''
(P/'REPORT.md').write_text(text)
print(json.dumps(summary,indent=2))
