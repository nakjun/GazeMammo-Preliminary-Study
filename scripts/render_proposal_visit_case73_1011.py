"""Scientific case73 figure from saved image-only traces; annotations are posthoc."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluate_proposal_policy_1011 import rectangles,coverage,distinct_proposals,lesion_metrics


def selected_traces(path,policies,image_id=73,seed=42):
    result={};last=-1
    with Path(path).open() as source:
        for line in source:
            record=json.loads(line);current=int(record['image_id'])
            if current<last:raise ValueError('Expected traces sorted by image_id for bounded streaming')
            last=current
            if current>image_id:break
            if current==image_id and int(record['seed'])==seed and record['policy'] in policies:
                if record['policy'] in result:raise ValueError('Duplicate selected trace')
                if record.get('annotations_used',record.get('annotation_used')) is not False:raise ValueError('Generation annotation flag must be false')
                result[record['policy']]={k:record[k] for k in ['image_id','image','image_size','grid','crop_size','policy','seed','coordinates','indices','relevance']}
                if len(result)==len(policies):break
    missing=set(policies)-set(result)
    if missing:raise ValueError(f'Missing requested case traces {missing} in {path}')
    return result


def render(run_root,output):
    run_root=Path(run_root);output=Path(output)
    original=selected_traces(run_root/'original_sampling/traces.jsonl',('P0','P4','P5'))
    utility=selected_traces(run_root/'utility_sampling/traces.jsonl',('P4_utility_mse','P5_utility_mse'))
    reference=original['P0'];size=np.array(reference['image_size']);gx,gy=reference['grid'];crop=reference['crop_size']
    annotations=json.loads((ROOT/'manifests/instances_clipped.json').read_text())['annotations'];boxes=[a['bbox'] for a in annotations if int(a['image_id'])==73]
    if not boxes:raise ValueError('Case73 posthoc lesion annotation missing')
    ids=np.arange(gx*gy);centers=np.stack([((ids%gx)+.5)/gx,((ids//gx)+.5)/gy],axis=1);candidate_rects=rectangles(centers,size,crop)
    with Image.open(reference['image']) as source:image=np.asarray(source.convert('L'))
    os.environ.setdefault('MPLCONFIGDIR','/tmp/proposal_mpl_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(2,4,figsize=(12,12),layout='constrained');axes=axes.ravel()
    def backdrop(axis):
        axis.imshow(image,cmap='gray',vmin=0,vmax=255)
        for x,y,w,h in boxes:axis.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='#ff4040',linewidth=1.8,zorder=8))
        axis.set(xlim=(0,size[0]),ylim=(size[1],0));axis.axis('off')
    for axis,label,trace in [(axes[0],'Original occupancy R',reference),(axes[1],'New crop utility U',utility['P4_utility_mse'])]:
        backdrop(axis);heat=np.asarray(trace['relevance']).reshape(gy,gx)
        im=axis.imshow(heat,cmap='magma',vmin=0,vmax=1,alpha=.7,extent=(0,size[0],size[1],0),interpolation='nearest')
        ranked=distinct_proposals(trace['relevance'],candidate_rects,k=3)
        for rank,index in enumerate(ranked):
            xy=centers[index]*size;axis.scatter(*xy,facecolors='none',edgecolors='#35d9ef',s=55,linewidths=1.2);axis.annotate(str(rank+1),xy,color='#35d9ef',fontsize=8,xytext=(5,4),textcoords='offset points')
        axis.set_title(label+'\nTop3 spatially distinct crops',fontsize=10)
    fig.colorbar(im,ax=list(axes[:2]),fraction=.04,pad=.01,label='Score (explicit 0–1; targets differ)')
    selected=[('Original P0',original['P0']),('Original P4',original['P4']),('Original P5',original['P5']),('Utility P4',utility['P4_utility_mse']),('Utility P5',utility['P5_utility_mse'])]
    diagnostics={}
    for axis,(label,trace) in zip(axes[2:],selected):
        if list(trace['image_size'])!=list(size) or trace['crop_size']!=224:raise ValueError('Inconsistent image/crop geometry')
        backdrop(axis);points=np.asarray(trace['coordinates']);pixels=points*size;rects=rectangles(points,size,crop)
        ranked=distinct_proposals(trace['relevance'],candidate_rects)
        metrics=[lesion_metrics(rects,candidate_rects,ranked,box) for box in boxes]
        recall=float(np.mean([m['recall_05_step16'] for m in metrics]));actual=float(np.mean([m['actual_box_coverage'] for m in metrics]))
        diagnostics[trace['policy']]=dict(seed=42,recall50=recall,actual_box_coverage=actual)
        axis.plot(pixels[:,0],pixels[:,1],color='#39d9ff',linewidth=.9,alpha=.7)
        axis.scatter(pixels[:,0],pixels[:,1],c=np.arange(1,len(points)+1),cmap='winter',vmin=1,vmax=16,s=18,zorder=5)
        # Show first three crops and all sufficient lesion-hit crops, without overcrowding every rectangle.
        visible=set(range(min(3,len(rects))))
        for box in boxes:visible.update(np.flatnonzero(coverage(rects,box)>=.5).tolist())
        for index in sorted(visible):
            x0,y0,x1,y1=rects[index];axis.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor='#39d9ff',edgecolor='#39d9ff',alpha=.16,linewidth=.6))
        for step in (1,4,8,16):
            if step<=len(points):axis.annotate(str(step),pixels[step-1],fontsize=7,color='white',xytext=(3,3),textcoords='offset points')
        axis.set_title(f'{label}, seed42\nRecall50 {recall:.0%}; box coverage {actual:.1%}',fontsize=10)
    axes[-1].axis('off');axes[-1].text(0,.95,'Case73: posthoc illustration\n\nRed: annotated lesion box\nCyan path: generated 16 steps\nFaint 224px rectangles:\nfirst 3 + sufficient-hit crops\n\nR and U target different geometry.\nScores are not calibrated\nmalignancy probabilities.\n\nAll 3 seed results belong in\ncase73_comparison.csv.\nExploratory validation case.',va='top',fontsize=10)
    fig.suptitle('Proposal detection and actual crop visits — case73',fontsize=14)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=180);plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(dict(image_id=73,seed=42,annotation_used_for_generation=False,posthoc_only=True,score_scale=[0,1],diagnostics=diagnostics),indent=2))
    return diagnostics

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',default='runs/proposal_visit_1011');parser.add_argument('--output');args=parser.parse_args();render(args.run_root,args.output or str(Path(args.run_root)/'case73_comparison.png'))
