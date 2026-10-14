"""Frozen image-only gaze feature extraction and binary diagnosis heads."""
import numpy as np
import torch
from torch import nn
from .data import whole_tensor, patch_tensor, select_centers, seed_for
from .trajectory_1011 import check_provenance


class GazeDiagnosisHead(nn.Module):
    def __init__(self, input_dim=1024, dim=128, mode='gaze', dropout=.3):
        super().__init__()
        if mode not in ('global', 'gaze', 'grid'):
            raise ValueError(mode)
        self.config = dict(input_dim=input_dim, dim=dim, mode=mode, dropout=dropout)
        self.mode = mode
        self.global_project = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim),
                                            nn.GELU(), nn.Dropout(dropout))
        if mode != 'global':
            self.local_project = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim), nn.GELU())
            self.position = nn.Linear(2, dim)
            self.sequence = nn.GRU(dim, dim, batch_first=True)
            self.attention = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim if mode == 'global' else 2 * dim, 1))

    def forward(self, global_features, local_features=None, coordinates=None):
        global_vector = self.global_project(global_features)
        if self.mode == 'global':
            return self.head(global_vector).squeeze(-1), None
        if local_features is None or coordinates is None:
            raise ValueError('Patch features and coordinates required')
        if local_features.shape[:2] != coordinates.shape[:2] or coordinates.shape[-1] != 2:
            raise ValueError('Patch/coordinate shape mismatch')
        x = self.local_project(local_features) + self.position(coordinates)
        x, _ = self.sequence(x)
        weights = self.attention(x).squeeze(-1).softmax(1)
        local_vector = (x * weights[..., None]).sum(1)
        return self.head(torch.cat([global_vector, local_vector], -1)).squeeze(-1), weights


def validate_population(frame, checkpoint):
    """Train reuse is transfer learning, never misrepresented as cross-fitting."""
    if frame.image_id.duplicated().any() or (frame.groupby('patient_key').split.nunique() > 1).any():
        raise ValueError('Duplicate IDs or patient split overlap')
    tr = frame[frame.split == 'train']
    va = frame[frame.split == 'val']
    check_provenance(checkpoint, set(tr.patient_key), set(va.patient_key), set(va.patient_key))
    selected = frame[frame.split.isin(['train', 'val']) & frame.label.isin([0, 1])].copy()
    if any(selected[selected.split == split].label.nunique() != 2 for split in ('train', 'val')):
        raise ValueError('Both classes required in training and validation')
    return selected.sort_values(['split', 'image_id']).reset_index(drop=True)


@torch.inference_mode()
def image_features(model, image, image_id, seed=42, steps=16, include_grid=True):
    """Predict from pixels and prior predictions only; no gaze/label/box reads."""
    if image.size != (1500, 3000):
        raise ValueError('Expected existing dataset preprocessing: 1500 x 3000')
    model.eval()
    device = next(model.parameters()).device
    whole = whole_tensor(image, model.input_size)[None].to(device)
    memory = model.maps(whole)
    global_features = model.visual.patch_vector(whole)[0]
    prefix = torch.empty(1, 0, 2, device=device)
    generator = torch.Generator(device='cpu').manual_seed(seed_for(seed, image_id))
    hidden = vector = None
    patches = []
    for _ in range(steps):
        logp, hidden, _ = model.step_history(memory, prefix, hidden, vector)
        # Use a CPU RNG for the same seed convention in extraction and inference.
        cell = int(torch.multinomial(logp.softmax(-1).cpu(), 1, generator=generator))
        point = model.cell_xy[cell]
        prefix = torch.cat([prefix, point[None, None]], 1)
        crop = patch_tensor(image, point.cpu().numpy() * image.size)[None].to(device)
        vector = model.visual.patch_vector(crop)
        patches.append(vector[0])
    result = dict(global_features=global_features.cpu(), gaze_features=torch.stack(patches).cpu(),
                  gaze_coordinates=prefix[0].cpu())
    if include_grid:
        centers = select_centers(image, 'grid', steps, seed_for(seed, image_id))
        if len(centers) != steps:
            raise ValueError('Insufficient foreground for grid comparison')
        vectors = []
        for start in range(0, steps, 8):
            crops = torch.stack([patch_tensor(image, p) for p in centers[start:start + 8]]).to(device)
            vectors.append(model.visual.patch_vector(crops).cpu())
        result.update(grid_features=torch.cat(vectors),
                      grid_coordinates=torch.tensor(centers / image.size, dtype=torch.float32))
    return result
