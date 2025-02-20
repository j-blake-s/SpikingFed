import torch
from torch import nn
# from .spikingV2 import SpkConv, SpkDense
import lava.lib.dl.slayer as slayer

# from spikingjelly.clock_driven.neuron import MultiStepLIFNode as LIF
params = {
  'threshold'     : 1,
  'current_decay' : 0.3,
  'voltage_decay' : 0.25,
  'tau_grad'      : 0.001,
  'requires_grad' : True,
}

def SpkConv(in_, out_, kernel_size=5, stride=4, padding=1):
  return slayer.block.cuba.Conv(
    params, in_, out_, kernel_size=kernel_size, stride=stride, 
    padding=padding, weight_scale=2, weight_norm=True, delay=False, 
    delay_shift=False,
  )
def SpkDense(in_, out_):
  return slayer.block.cuba.Dense(
    params, in_, out_, weight_scale=2, weight_norm=True, delay_shift=False,
  )

class SNN(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv = nn.ModuleList([
      SpkConv(2, 8),
      SpkConv(8, 32),
      SpkConv(32, 64),
    ])

    self.dense = nn.ModuleList([
      SpkDense(64*2*2, 2056),
      SpkDense(2056, 11),
    ])


  def params(self):
    model_ps = filter(lambda p: p.requires_grad, self.parameters())
    p_count = sum([torch.prod(torch.tensor(p.size())) for p in model_ps])
    return p_count.item()
  
  def forward(self, x):
     
    B, C, H, W, T = x.shape
    
    # Conv
    for c in self.conv:
      x = c(x)

    # Flatten
    x = x.flatten(1,3)

    # Dense
    for d in self.dense:
      x = d(x)

    return x
#######################################
  def load_weights(self, state_dict):
        """Loads new weights into the SNN model during FL aggregation."""
        self.load_state_dict(state_dict)
#######################################