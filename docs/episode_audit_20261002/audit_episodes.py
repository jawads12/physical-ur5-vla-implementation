from pathlib import Path
import concurrent.futures,csv,json,sys,time,hashlib
import numpy as np
from PIL import Image,ImageDraw
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'ros2_ws/src/ur5-vla-dataset-recorder'))
from ur5_vla_dataset_recorder.raw_dataset import validate_raw_episode
from ur5_vla_dataset_recorder.standard_pose_node import STANDARD_POSE_RAD
ROOT=PROJECT/'data/dataset/staging/episodes'
OUT=Path(__file__).resolve().parent
BASE=PROJECT/'data/dataset/lerobot/ur5_cube_clean_73ep_v2/meta/ur5_vla_conversion.json'
TASK='Pick up the orange cube and place it in the bowl'

def structural(name):
 p=ROOT/name;v=validate_raw_episode(p,decode_all_media=False);f=v.frames
 t=np.asarray([x['episode_time_s'] for x in f]);a=np.asarray([x['action'] for x in f]);s=np.asarray([x['observation.state'] for x in f])
 transitions=(np.flatnonzero(np.diff(a[:,6])!=0)+1).tolist()
 dt=np.diff(t);steps=np.max(np.abs(np.diff(a[:,:6],axis=0)),axis=1)
 notes=[]
 if len(transitions)!=2 or a[0,6]!=1 or a[-1,6]!=1:notes.append('not a single open-close-open sequence')
 start_error=float(np.max(np.abs(s[0,:6]-STANDARD_POSE_RAD)))
 if start_error>.05:notes.append('initial joints differ from standard pose by >0.05 rad')
 tasks=sorted({x.get('task','') for x in f})
 if tasks!=[TASK] or v.metadata.get('task')!=TASK:notes.append('task text mismatch')
 extras={'non_increasing_camera_timestamps':{},'longest_unchanged_image_timestamp_s':{}}
 for cam in ['base','wrist']:
  stamps=np.asarray([x.get(cam+'_image_source_stamp_ns',0) for x in f],dtype=np.int64)
  extras['non_increasing_camera_timestamps'][cam]=int(np.count_nonzero(np.diff(stamps)<0))
  changes=np.r_[0,np.flatnonzero(np.diff(stamps)!=0)+1,len(stamps)]
  maxhold=max(float(t[end-1]-t[start]) for start,end in zip(changes[:-1],changes[1:]))
  extras['longest_unchanged_image_timestamp_s'][cam]=maxhold
  if maxhold>.2:notes.append(cam+' image timestamp unchanged for >0.2s')
  if extras['non_increasing_camera_timestamps'][cam]:notes.append(cam+' image timestamps move backwards')
 skew=np.abs(np.asarray([x['base_image_source_stamp_ns']-x['wrist_image_source_stamp_ns'] for x in f],dtype=np.float64))/1e9
 r={'episode':name,'frames':len(f),'duration_s':float(t[-1]-t[0]),'dropped_frames':v.metadata.get('dropped_frames'),
 'structural_pass':v.ok,'errors':v.errors,'warnings':v.warnings,'review_notes':notes,'transitions':[{'frame':i,'time_s':float(t[i]),'to':int(a[i,6])} for i in transitions],
 'max_gap_s':float(np.max(dt)),'gaps_over_0_2s':int(np.count_nonzero(dt>.2)),'max_action_step_rad':float(np.max(steps)),
 'max_state_action_error_rad':float(np.max(np.abs(a[:,:6]-s[:,:6]))),'start_pose_error_rad':start_error,'camera_skew_p99_s':float(np.quantile(skew,.99)),
 'metadata_sha256':hashlib.sha256((p/'metadata.yaml').read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((p/'frames.jsonl').read_bytes()).hexdigest(),**extras}
 # Four snapshots, both camera views: this is a contact sheet, not full video review.
 indices=[0,transitions[0] if transitions else len(f)//3,transitions[-1] if transitions else 2*len(f)//3,len(f)-1]
 sheet=Image.new('RGB',(192*8,170),'white');draw=ImageDraw.Draw(sheet)
 for k,idx in enumerate(indices):
  for c,cam in enumerate(['base','wrist']):
   col=2*k+c
   with Image.open(p/f[idx]['observation.images.'+cam]) as im:sheet.paste(im.convert('RGB').resize((192,144)),(col*192,26))
   draw.text((col*192+3,3),f'{cam} f{idx} t={t[idx]:.1f}',fill='black')
 (OUT/'snapshots').mkdir(exist_ok=True);sheet.save(OUT/'snapshots'/f'{name}.jpg',quality=90)
 return r

def full(name):
 start=time.monotonic();v=validate_raw_episode(ROOT/name,decode_all_media=True)
 return {'episode':name,'full_media_pass':v.ok,'frames':len(v.frames),'errors':v.errors,'warnings':v.warnings,'seconds':round(time.monotonic()-start,2)}

if __name__=='__main__':
 names=sorted(p.name for p in ROOT.glob('episode_*') if p.is_dir() and p.name not in set(json.loads(BASE.read_text())['source_episode_names']))
 (OUT/'new_episodes.txt').write_text('\n'.join(names)+'\n')
 summary=[]
 for i,name in enumerate(names,1):
  r=structural(name);summary.append(r)
  print('STRUCTURAL',i,len(names),name,'PASS' if r['structural_pass'] else 'FAIL',r['errors'],r['review_notes'],flush=True)
 (OUT/'structural_report.json').write_text(json.dumps(summary,indent=2)+'\n')
 for start in range(0,len(names),8):
  chunk=names[start:start+8];page=Image.new('RGB',(1536,194*len(chunk)),'white');draw=ImageDraw.Draw(page)
  for row,name in enumerate(chunk):
   draw.text((4,row*194+2),name,fill='black')
   with Image.open(OUT/'snapshots'/f'{name}.jpg') as im:page.paste(im,(0,row*194+24))
  page.save(OUT/f'contact_sheet_{start//8+1}.jpg',quality=92)
 print('FULL MEDIA CHECK START',flush=True)
 completed=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  pending={pool.submit(full,name):name for name in names}
  for future in concurrent.futures.as_completed(pending):
   r=future.result();completed.append(r)
   (OUT/'full_media_report.json').write_text(json.dumps(sorted(completed,key=lambda x:x['episode']),indent=2)+'\n')
   print('FULL',len(completed),len(names),r['episode'],'PASS' if r['full_media_pass'] else 'FAIL',r['seconds'],r['errors'][:8],flush=True)
 print('AUDIT COMPLETE',flush=True)
