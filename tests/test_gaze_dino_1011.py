"""Actual pretrained frozen DINO causal head and strict compact restore."""
import json
from pathlib import Path
import torch
from src.gaze_dino_1011 import DinoGaze,restore_dino
from src.guided_trajectory_1011 import nonvisual_state
from src.proposal_utility_1011 import state_sha256

def test_actual_causal_restore():
 torch.set_num_threads(4);torch.manual_seed(42);source=json.loads(Path('runs/dual_target_1011/dino/source_provenance.json').read_text());model=DinoGaze(source['source'],source['checkpoint']).eval();r=torch.load('runs/dual_target_1011/dino/smoke64_73/64.pt',weights_only=False);f=r['feature'][None];coords=r['coords'][None];vectors=r['vectors'][None];memory=model.maps_features([f]*3);raw=model.teacher_logits(memory,coords,vectors);changed=coords.clone();changed[:,8:]=.9;modified=vectors.clone();modified[:,9:]=torch.randn_like(modified[:,9:]);assert torch.equal(raw[:,:9],model.teacher_logits(memory,changed,modified)[:,:9]);ck=dict(kind='dino_gaze_1011',state=nonvisual_state(model),config=dict(source=source),visual_state_sha256=state_sha256(model.visual.state_dict()));restored=restore_dino(ck);assert torch.equal(raw,restored.teacher_logits(restored.maps_features([f]*3),coords,vectors));(-raw[:,:,100].mean()).backward();assert all(p.grad is None and not p.requires_grad for p in model.visual.parameters());assert torch.isfinite(model.density.weight.grad).all()
