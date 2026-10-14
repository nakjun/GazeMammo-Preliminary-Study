"""Image-only representative centers from connected relative-score proposal regions."""
import numpy as np
from scipy import ndimage


def snap_lower_tie(value):
    """Nearest grid coordinate; exact .5 ties choose the smaller coordinate."""
    return int(np.floor(float(value)+.5-1e-12))


def crop_iou(a,b):
    intersection=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(intersection/(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection))


def region_candidate_indices(scores,grid=(24,48),mode='peak',threshold=.25,window=7,rectangles=None,budget=None,max_iou=.3,return_regions=False):
    """No annotations or labels. Connected8 regions are bounded to each peak's window.

    peak preserves original all-cell score ranking/NMS. New region modes rank local
    peaks by original peak score, recenter, snap, then remove duplicates and NMS.
    """
    gx,gy=grid
    if hasattr(scores,'detach'):scores=scores.detach().cpu().numpy()
    values=np.asarray(scores,dtype=float).reshape(-1)
    if values.shape!=(gx*gy,) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):raise ValueError('Expected finite sigmoid proposal grid scores')
    if mode not in ('peak','centroid','bbox_center') or threshold not in (.25,.5) or window<1 or window%2!=1 or not 0<=max_iou<=1:raise ValueError('Invalid region protocol')
    if budget is not None and budget<0:raise ValueError('Invalid budget')
    if rectangles is not None:
        if hasattr(rectangles,'detach'):rectangles=rectangles.detach().cpu().numpy()
        rectangles=np.asarray(rectangles,dtype=float)
        if rectangles.shape!=(gx*gy,4):raise ValueError('Invalid candidate crop rectangles')
    score_map=values.reshape(gy,gx);regions=[]
    if mode=='peak':
        peaks=np.argsort(-values,kind='stable')
    else:
        is_peak=score_map==ndimage.maximum_filter(score_map,size=3,mode='constant',cval=-np.inf)
        labels,count=ndimage.label(is_peak,structure=np.ones((3,3)))
        # A connected equal-score plateau supplies one stable, smallest-index peak.
        peaks=[int(np.flatnonzero(labels.reshape(-1)==label)[0]) for label in range(1,count+1)]
        peaks=sorted(peaks,key=lambda i:(-values[i],i))
    radius=window//2
    for peak in peaks:
        py,px=divmod(int(peak),gx)
        representative=int(peak);component=[];centroid=[float(px),float(py)];box_center=centroid[:]
        if mode!='peak':
            x0,x1=max(0,px-radius),min(gx,px+radius+1);y0,y1=max(0,py-radius),min(gy,py+radius+1)
            patch=score_map[y0:y1,x0:x1];active=patch>=threshold*values[peak]
            labels,_=ndimage.label(active,structure=np.ones((3,3)));label=int(labels[py-y0,px-x0]);yy,xx=np.where(labels==label);xx=xx+x0;yy=yy+y0
            weights=score_map[yy,xx]**2
            if weights.sum()>0:centroid=[float(np.average(xx,weights=weights)),float(np.average(yy,weights=weights))]
            box_center=[float((xx.min()+xx.max())/2),float((yy.min()+yy.max())/2)]
            center=centroid if mode=='centroid' else box_center
            cx=int(np.clip(snap_lower_tie(center[0]),0,gx-1));cy=int(np.clip(snap_lower_tie(center[1]),0,gy-1));representative=cy*gx+cx;component=(yy*gx+xx).tolist()
        regions.append(dict(peak_index=int(peak),peak_score=float(values[peak]),representative_index=representative,component_indices=component,weighted_centroid_xy=centroid,bbox_center_xy=box_center))
    selected=[];selected_regions=[]
    for region in regions:
        index=region['representative_index']
        if index in selected:continue
        if rectangles is not None and any(crop_iou(rectangles[index],rectangles[previous])>max_iou for previous in selected):continue
        selected.append(index);selected_regions.append(region)
        if budget is not None and len(selected)>=budget:break
    if budget==0:selected=[];selected_regions=[]
    return (selected,selected_regions) if return_regions else selected
