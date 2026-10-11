# Raw Episode Validation Results

Validated on 2026-08-21 with `validate_raw_episode(..., decode_all_media=True)` before conversion. The validator parsed every manifest row and decoded every referenced RGB and uint16 depth file.

| Episode | Frames | Result |
|---|---:|---|
| `episode_20260821_012008` | 2,409 | PASS |
| `episode_20260821_013856` | 3,085 | PASS |
| `episode_20260821_014409` | 2,709 | PASS |
| `episode_20260821_015256` | 1,651 | PASS |
| `episode_20260821_015504` | 1,733 | PASS |
| `episode_20260821_015723` | 1,960 | PASS |
| `episode_20260821_020024` | 2,299 | PASS |
| `episode_20260821_020621` | 2,239 | PASS |
| `episode_20260821_021553` | 1,600 | PASS |
| `episode_20260821_021830` | 898 | PASS |
| **Total** | **20,583** | **10/10 PASS** |

Each pass confirms finalized `success` metadata, canonical state/command joint order, metadata/manifest count agreement, sequential indexes, strictly increasing finite timestamps, finite 7D state/action, binary consistent gripper fields, at least one close and open transition, `valid: true`, Servo status zero, freshness within 0.10 s (camera/depth/joints/action) and 0.20 s (gripper), arm action jumps at or below 0.20 rad/frame, complete one-to-one media references/counts, decodable 640×480 RGB and uint16 depth, and distinct sampled base/wrist views.

Hidden/incomplete directories, editor settings, prior validation artifacts, and virtual environments were excluded from episode selection. Raw data was not modified.
