"""Durable combined full training and explicitly post-hoc fixed-case checks."""
import datetime,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/guided_1011_combined_seed42'
MONITOR=ROOT/'runs/guided_1011_combined_seed42_monitor'


def save(name,value):
    temporary=MONITOR/(name+'.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(MONITOR/name)

def execute(stage,args):
    command=[sys.executable,'-u',str(ROOT/'scripts/guided_trajectory_1011.py'),*args]
    log=MONITOR/(stage+'.log');environment=os.environ.copy();environment['CUDA_VISIBLE_DEVICES']='0';environment['PYTHONUNBUFFERED']='1'
    with log.open('x') as output:
        process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=output,stderr=subprocess.STDOUT)
        save('status.json',dict(stage=stage,state='running',pid=process.pid,wrapper_pid=os.getpid(),gpu=0,tmux='guided_full_1011',command=command,log=str(log),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        code=process.wait()
    if code:
        save('status.json',dict(stage=stage,state='failed',returncode=code,pid=process.pid,log=str(log)))
        raise RuntimeError(f'{stage} failed with code{code}; see {log}')


def posthoc_overlay(image_id,directory):
    import os
    os.environ.setdefault('MPLCONFIGDIR','/tmp/gaze_mammo_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    import numpy as np
    record=json.loads((directory/'prediction.json').read_text());metrics=json.loads((directory/'posthoc_box_metrics.json').read_text())
    with Image.open(record['image']) as source:image=source.convert('L')
    points=np.asarray(record['coordinates'])*image.size;fig,axis=plt.subplots(figsize=(6,11));axis.imshow(image,cmap='gray',vmin=0,vmax=255);axis.plot(*points.T,color='cyan',linewidth=1);axis.scatter(*points.T,c=np.arange(len(points)),cmap='turbo',s=20)
    for i,(x,y) in enumerate(points):axis.annotate(str(i+1),(x,y),color='cyan',fontsize=7)
    for x,y,w,h in metrics['lesion_boxes']:axis.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='red',linewidth=1.5))
    axis.set(xlim=(0,image.width),ylim=(image.height,0),title=f'Case{image_id}, selected guided epoch{record["checkpoint_epoch"]}\nRed GT boxes drawn after image-only prediction');axis.axis('off');fig.tight_layout();fig.savefig(directory/'posthoc_gt_box_comparison.png',dpi=160);plt.close(fig)


def main():
    MONITOR.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    execute('train',['train','--ablation','combined','--epochs','15','--device','cuda:0','--seed','42','--train-limit','0','--val-limit','0','--visual-every','3','--threads','4','--output',str(RUN),'--base-checkpoint',str(ROOT/'runs/full_1011_seed42/improved/best.pt')])
    import pandas as pd
    manifest=pd.read_csv(ROOT/'manifests/all.csv')
    for image_id in (64,73):
        row=manifest[manifest.image_id==image_id].iloc[0];directory=RUN/f'best_predict_case{image_id}'
        execute(f'predict_case{image_id}',['predict','--checkpoint',str(RUN/'best.pt'),'--image',str(ROOT/'data/image'/row.file_name),'--image-id',str(image_id),'--seed','42','--device','cuda:0','--threads','4','--output',str(directory),'--posthoc-annotations',str(ROOT/'manifests/instances_clipped.json')])
        posthoc_overlay(image_id,directory)
    save('status.json',dict(state='complete',stage='complete',seconds=time.monotonic()-started,run=str(RUN),tmux='guided_full_1011',wrapper_pid=os.getpid()))

if __name__=='__main__':main()
