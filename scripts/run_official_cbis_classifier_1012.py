#!/usr/bin/env python
"""Strict official CBIS classification stages; TEST requires all 15 heads frozen."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.official_cbis_classifier_1012 import build_cache,train_all,freeze_all,evaluate_test

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['cache','train','freeze','evaluate'])
    parser.add_argument('--metadata',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--baseline');parser.add_argument('--improved');parser.add_argument('--cache');parser.add_argument('--heads');parser.add_argument('--freeze')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--split',choices=['trainval','test'],default='trainval');parser.add_argument('--device',default='cuda:0');parser.add_argument('--epochs',type=int,default=60);parser.add_argument('--patience',type=int,default=12);parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    required={'cache':['baseline','improved'],'train':['cache'],'freeze':['baseline','improved','heads'],'evaluate':['cache','freeze']}
    for key in required[args.phase]:
        if getattr(args,key) is None:parser.error('--'+key+' required for '+args.phase)
    if args.phase=='cache':build_cache(args.metadata,args.baseline,args.improved,args.output,args.split,args.device,args.freeze,args.smoke)
    elif args.phase=='train':train_all(args.cache,args.metadata,args.output,args.epochs,args.patience)
    elif args.phase=='freeze':freeze_all(args.metadata,args.heads,args.baseline,args.improved,args.output)
    else:evaluate_test(args.cache,args.metadata,args.freeze,args.output)
if __name__=='__main__':main()
