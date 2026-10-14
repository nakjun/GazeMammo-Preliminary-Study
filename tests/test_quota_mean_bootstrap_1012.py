import numpy as np
from sklearn.metrics import roc_auc_score
from scripts.confirm_gaze_quota_three_seed_1012 import WeightedAUC

def test_weighted_cluster_auc_matches_expanded_tied_auc():
 y=np.array([0,1,0,1,1,0]);p=np.array([.2,.4,.4,.9,.4,.9]);w=np.array([2,3,1,2,1,4]);ids=np.repeat(np.arange(len(y)),w);assert abs(WeightedAUC(y,p)(w)-roc_auc_score(y[ids],p[ids]))<1e-14
 assert abs(WeightedAUC(y,p)(np.ones(len(y)))-roc_auc_score(y,p))<1e-14
if __name__=='__main__':
 test_weighted_cluster_auc_matches_expanded_tied_auc();print('WeightedpatientbootstrapAUC tie/repeatedrowparity passed')
