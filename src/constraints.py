"""Image-only breast masks and constrained autoregressive trajectory decoding."""
import numpy as np
from scipy import ndimage
import torch

def breast_mask(image,threshold=10):
    """Largest bright component + hole filling; an estimated mask, not a GT segmentation."""
    a=np.asarray(image.convert('L'))
    components,n=ndimage.label(a>threshold)
    if n==0:raise ValueError('No breast foreground found')
    counts=np.bincount(components.ravel());counts[0]=0
    mask=components==counts.argmax()
    mask=ndimage.binary_fill_holes(mask)
    # Pad so closing does not erase tissue at the acquisition boundary.
    mask=ndimage.binary_closing(np.pad(mask,3,mode='edge'),iterations=3)[3:-3,3:-3]
    return mask

def grid_mask(mask,grid):
    gx,gy=grid;h,w=mask.shape
    x=np.minimum(((np.arange(gx)+.5)/gx*w).astype(int),w-1)
    y=np.minimum(((np.arange(gy)+.5)/gy*h).astype(int),h-1)
    return mask[y[:,None],x[None,:]].reshape(-1)

class VisitConstraint:
    def __init__(self,allowed,sizes,grid,kind='breast',radius=112,penalty=3.):
        if kind not in ('none','breast','hard','soft'):raise ValueError(kind)
        if radius<0 or penalty<0:raise ValueError('Radius and penalty must be nonnegative')
        self.allowed=allowed.bool();self.sizes=sizes;self.kind=kind;self.radius=radius;self.penalty=penalty
        gx,gy=grid;ids=torch.arange(gx*gy,device=allowed.device)
        self.coords=torch.stack([((ids%gx)+.5)/gx,((ids//gx)+.5)/gy],-1)
        self.visited=torch.zeros_like(self.allowed)
        self.closest=torch.full(self.allowed.shape,torch.inf,device=allowed.device)
        self.active=torch.ones(len(allowed),dtype=torch.bool,device=allowed.device)
        self.relaxations=torch.zeros(len(allowed),dtype=torch.long,device=allowed.device)
    def apply(self,logits):
        score=logits.clone();allowed=torch.ones_like(self.allowed) if self.kind=='none' else self.allowed.clone()
        if self.kind=='soft' and self.radius>0:
            score-=self.penalty*torch.exp(-self.closest.square()/(2*self.radius**2))
        if self.kind=='hard':
            unvisited=allowed & ~self.visited
            radius=torch.full((len(score),1),float(self.radius),device=score.device)
            allowed=unvisited & (self.closest>=radius)
            missing=(~allowed.any(1)) & unvisited.any(1) & self.active
            relaxed=missing.clone()
            # Relax distance, never the breast mask or the exact-cell exclusion.
            for _ in range(32):
                if not missing.any():break
                radius[missing]*=.5
                radius[radius<1e-4]=0
                allowed=unvisited & (self.closest>=radius)
                missing=(~allowed.any(1)) & unvisited.any(1) & self.active
            self.relaxations+=relaxed.long()
        self.active &= allowed.any(1)
        # Finished images receive an inert finite token; valid flags exclude it from outputs.
        allowed[~self.active,0]=True
        return score.masked_fill(~allowed,-torch.inf),self.active.clone()
    def update(self,indices,valid):
        rows=torch.arange(len(indices),device=indices.device)
        self.visited[rows[valid],indices[valid]]=True
        distances=torch.linalg.vector_norm((self.coords[None]-self.coords[indices,None])*self.sizes[:,None],dim=-1)
        self.closest=torch.where(valid[:,None],torch.minimum(self.closest,distances),self.closest)

@torch.no_grad()
def rollout(model,memory,allowed,sizes,kind='none',radius=112,penalty=3.,strategy='argmax',temperature=1.):
    if strategy not in ('argmax','sample') or temperature<=0:raise ValueError('Invalid decoding configuration')
    state=VisitConstraint(allowed,sizes,model.grid,kind,radius,penalty)
    tokens=torch.full((len(memory),1),model.cells,device=memory.device,dtype=torch.long)
    generated=[];validity=[]
    for _ in range(model.k):
        logits,valid=state.apply(model.decode(memory,tokens)[:,-1])
        next_token=logits.argmax(-1) if strategy=='argmax' else torch.multinomial((logits/temperature).softmax(-1),1).squeeze(-1)
        state.update(next_token,valid);generated.append(next_token);validity.append(valid)
        tokens=torch.cat([tokens,next_token[:,None]],1)
    ids=torch.stack(generated,1)
    return dict(coords=state.coords[ids],valid=torch.stack(validity,1),relaxations=state.relaxations)

def spatial_metrics(points,mask,boxes,patch_size=224):
    h,w=mask.shape;p=np.asarray(points)*[w,h]
    indices=np.clip(p.astype(int),[0,0],[w-1,h-1]);inside=mask[indices[:,1],indices[:,0]]
    distances=np.linalg.norm(p[:,None]-p[None,:],axis=-1);pairs=np.triu_indices(len(p),1)
    rect=np.c_[p-patch_size/2,p+patch_size/2]
    overlap=[]
    for i,j in zip(*pairs):
        area=np.maximum(0,np.minimum(rect[i,2:],rect[j,2:])-np.maximum(rect[i,:2],rect[j,:2])).prod()
        overlap.append(float(area/patch_size**2))
    union=np.zeros_like(mask);fractions=[]
    for x0,y0,x1,y1 in rect:
        x0,y0=max(0,int(round(x0))),max(0,int(round(y0)))
        x1,y1=min(w,int(round(x1))),min(h,int(round(y1)))
        union[y0:y1,x0:x1]=True;fractions.append(float(mask[y0:y1,x0:x1].sum()/patch_size**2))
    hits=[];coverage=[]
    for x,y,bw,bh in boxes:
        area=np.maximum(0,np.minimum(rect[:,2],x+bw)-np.maximum(rect[:,0],x))*np.maximum(0,np.minimum(rect[:,3],y+bh)-np.maximum(rect[:,1],y))
        hits.append(bool((area>0).any()));coverage.append(float(area.max()/(bw*bh)))
    return dict(n_points=len(p),unique_locations=len(np.unique(p,axis=0)),out_of_breast_fraction=float((~inside).mean()),
                mean_pairwise_patch_overlap=float(np.mean(overlap)) if overlap else 0.,
                minimum_center_distance=float(distances[pairs].min()) if len(p)>1 else None,
                mean_patch_breast_fraction=float(np.mean(fractions)),breast_area_covered=float((union&mask).sum()/mask.sum()),
                any_lesion_hit=bool(any(hits)),lesion_hit_rate=float(np.mean(hits)) if hits else None,
                best_patch_lesion_coverage=float(np.mean(coverage)) if coverage else None)
