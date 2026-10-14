import torch
from src.gaze_path_augmentation_1011 import augmented_sample

def test_actual_control_dino_augmented_prefixes_and_previous_vectors_match():
    torch.set_num_threads(4)
    for image_id in [62,65]:
        control=torch.load(f'runs/dual_target_1011/train_gaze_pool_cache/{image_id}.pt',weights_only=False);dino=torch.load(f'runs/dual_target_1011/dino/pool_cache/{image_id}.pt',weights_only=False)
        a=augmented_sample(control,image_id,2,42);b=augmented_sample(dino,image_id,2,42);assert a['pool_indices']==b['pool_indices'] and a['raw_indices']==b['raw_indices'];assert torch.equal(a['coords'],b['coords']) and torch.equal(a['mask'],b['mask']);assert a['vectors'].shape==(16,1024) and b['vectors'].shape==(16,384)
        for sample,pool in [(a,control),(b,dino)]:
            assert not sample['vectors'][0].any()
            assert torch.equal(sample['vectors'][1:],pool['patch_vectors'][sample['pool_indices'][:-1]])

def test_forward_kl_gradient_is_student_minus_detached_teacher():
    logits=torch.tensor([[.3,-.2,.7]],requires_grad=True);teacher=torch.tensor([[.2,.3,.5]]);raw=logits.log_softmax(-1);loss=(teacher*(teacher.log()-raw)).sum();loss.backward();assert torch.allclose(logits.grad,raw.detach().exp()-teacher,atol=1e-7)
