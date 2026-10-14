import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix,brier_score_loss

def threshold_at_specificity(labels,probs,target=.9):
    negative=np.asarray(probs)[np.asarray(labels)==0]
    if not len(negative):raise ValueError('Validation lacks negative samples')
    candidates=np.r_[np.unique(negative),np.nextafter(negative.max(),np.inf)]
    return float(next(t for t in candidates if (negative<t).mean()>=target))

def classification_metrics(labels,probs,threshold=.5):
    y=np.asarray(labels,dtype=int);p=np.asarray(probs);pred=p>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    sensitivity=float(tp/(tp+fn)) if tp+fn else None
    specificity=float(tn/(tn+fp)) if tn+fp else None
    return dict(n=len(y),auroc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
                auprc=float(average_precision_score(y,p)) if y.sum() else None,
                sensitivity=sensitivity,specificity=specificity,
                balanced_accuracy=(sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,
                brier=float(brier_score_loss(y,p)),threshold=float(threshold),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def cluster_bootstrap(frame,other=None,repetitions=1000,seed=42):
    rng=np.random.default_rng(seed);groups=frame.patient_key.unique();values=[]
    by_group={g:frame.index[frame.patient_key==g].to_numpy() for g in groups}
    for _ in range(repetitions):
        ids=np.concatenate([by_group[g] for g in rng.choice(groups,len(groups),replace=True)])
        sample=frame.loc[ids]
        if sample.label.nunique()!=2:continue
        score=roc_auc_score(sample.label,sample.probability)
        if other is not None:
            aligned=other.set_index('image_id').loc[sample.image_id]
            score-=roc_auc_score(sample.label,aligned.probability)
        values.append(score)
    if not values:return {'low':None,'high':None,'valid_replicates':0}
    return dict(low=float(np.quantile(values,.025)),high=float(np.quantile(values,.975)),valid_replicates=len(values))

def dtw_distance(a,b):
    table=np.full((len(a)+1,len(b)+1),np.inf);table[0,0]=0
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            table[i,j]=np.linalg.norm(a[i-1]-b[j-1])+min(table[i-1,j],table[i,j-1],table[i-1,j-1])
    return float(table[-1,-1]/(len(a)+len(b)))
