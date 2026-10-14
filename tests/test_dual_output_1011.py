import unittest,tempfile
from pathlib import Path
import torch
from torch import nn
from src.dual_output_1011 import ProbabilityComponent,config_sha256,validate_dependencies
from src.proposal_utility_1011 import file_sha256

class Visual(nn.Module):
    def __init__(self,dim):super().__init__();self.dim=dim
    def patch_vector(self,x):return x.mean((1,2,3))[:,None].expand(-1,self.dim)
class Parent(nn.Module):
    grid=(2,2);input_size=(10,10);foveal=True
    def __init__(self,dim,bias):
        super().__init__();self.register_buffer('cell_xy',torch.tensor([[.25,.25],[.75,.25],[.25,.75],[.75,.75]]));self.patch=nn.Linear(dim,1);self.visual=Visual(dim);self.bias=bias;self.register_buffer('pattern',torch.arange(4.).flip(0) if bias else torch.arange(4.))
    def maps(self,x):return dict(prior=torch.zeros(len(x),4))
    def step_history(self,memory,prefix,hidden=None,patch_vector=None):
        state=torch.zeros(len(prefix),1) if hidden is None else hidden
        previous=prefix.sum((1,2))[:,None];v=patch_vector.sum(-1,keepdim=True) if patch_vector is not None else state*0;state=state+1+self.bias;logits=self.pattern[None]+(previous+v+state)*torch.tensor([[.1,.2,.3,.4]])
        return logits.log_softmax(-1),state,{}

class DualTests(unittest.TestCase):
    def test_normalized_arithmetic_and_states(self):
        a,b=Parent(2,0),Parent(3,2);m=ProbabilityComponent([a,b],[.75,.25]);mem=m.maps(torch.zeros(1,3,10,10));prefix=torch.zeros(1,0,2);vector=torch.tensor([[1.,2.,3.,4.,5.]]);q,state,_=m.step_history(mem,prefix,patch_vector=vector);qa,_,_=a.step_history(mem['parents'][0],prefix,patch_vector=vector[:,:2]);qb,_,_=b.step_history(mem['parents'][1],prefix,patch_vector=vector[:,2:]);expected=.75*qa.exp()+.25*qb.exp();self.assertTrue(torch.allclose(q.exp(),expected));self.assertTrue(torch.allclose(q.exp().sum(-1),torch.ones(1)));self.assertEqual(float(state[0]),1);self.assertEqual(float(state[1]),3)
    def test_causal_teacher_and_vector_partition(self):
        m=ProbabilityComponent([Parent(2,0),Parent(3,2)],[.75,.25]);mem=m.maps(torch.zeros(1,3,10,10));coords=torch.rand(1,6,2);vectors=torch.rand(1,6,5);q=m.teacher_logits(mem,coords,vectors);changed=coords.clone();changed[:,3:]=.99;changed_v=vectors.clone();changed_v[:,4:]=100;q2=m.teacher_logits(mem,changed,changed_v);self.assertTrue(torch.equal(q[:,:4],q2[:,:4]));state=None;steps=[]
        for t in range(6):p,state,_=m.step_history(mem,coords[:,:t],state,vectors[:,t]);steps.append(p)
        self.assertTrue(torch.equal(q,torch.stack(steps,1)))
    def test_single_parent_and_restore(self):
        parent=Parent(2,0);m=ProbabilityComponent([parent],[1]);mem=m.maps(torch.zeros(1,3,10,10));coords=torch.rand(1,4,2);q=m.teacher_logits(mem,coords,torch.zeros(1,4,2))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.pt';torch.save(m.state_dict(),path);other=ProbabilityComponent([Parent(2,0)],[1]);other.load_state_dict(torch.load(path,weights_only=True));self.assertTrue(torch.equal(q,other.teacher_logits(other.maps(torch.zeros(1,3,10,10)),coords,torch.zeros(1,4,2))))
    def test_dependency_hash_and_config_identity(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'source.pt';path.write_bytes(b'original');dependency=dict(path=str(path),sha256=file_sha256(path));config=dict(kind='dual_output_1011',probability=dict(temperature=1,geometry_coefficient=0,parents=[dependency]),roi=dict(guided=dependency,offset=dependency));validate_dependencies(config);self.assertEqual(config_sha256(config),config_sha256(dict(reversed(list(config.items())))));path.write_bytes(b'changed')
            with self.assertRaises(ValueError):validate_dependencies(config)

if __name__=='__main__':unittest.main()
