










import torch
from torch import nn

  
class TConv2D(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
    super().__init__()
    self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

  def forward(self, x):
    B, C, H, W, T = x.shape

    x = torch.permute(x, [4,0,1,2,3]).flatten(0,1)
    x = self.conv(x)
    _, C, H, W = x.shape
    x = torch.permute(x.reshape(T, B, C, H, W), [1,2,3,4,0])

    return x

class TLinear(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.dense = nn.Linear(in_features=in_features, out_features=out_features)

  def forward(self, x):
    B, N, T = x.shape

    x = torch.permute(x, [2,0,1]).flatten(0,1)

    x = self.dense(x)
    _, N = x.shape
    x = torch.permute(x.reshape(T, B, N), [1,2,0])
    
    return x

class ANN(nn.Module):
  def __init__(self, args):
    super().__init__()

    self.conv = nn.ModuleList([
      TConv2D(2,   4),
      TConv2D(4,   8),
      TConv2D(8,  16),
      TConv2D(16, 32),
      TConv2D(32, 64),
      TConv2D(64, 64),
    ])

    self.dense = nn.ModuleList([
      TLinear(2*2*64, 512),
      TLinear(512, 64),
    ])
    
    self.out = TLinear(64, args.classes)


  def params(self):
    model_ps = filter(lambda p: p.requires_grad, self.parameters())
    p_count = sum([torch.prod(torch.tensor(p.size())) for p in model_ps])
    return p_count.item()

  def forward(self, x):
    B, C, H, W, T = x.shape

    for c in self.conv:
      x = c(x)
      x = nn.functional.relu(x)

    x = x.flatten(1,3)

    for d in self.dense:
      x = d(x)
      x = nn.functional.relu(x)
    
    x = self.out(x).mean(-1)
    
    if self.training:
      x = nn.functional.softmax(x, dim=-1)
    
    return x