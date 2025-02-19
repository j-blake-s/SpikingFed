import os
import numpy as np
from torch.utils.data import Dataset

# import cupy as cp
# def augment(x):
#   cx = cp.asarray(x)
#   # cx = h_flip(cx)
#   cx = temporal_jitter(cx, max_shift=4)
#   cx = spatial_jitter(cx, max_shift=20)
#   return cp.asnumpy(cx)


# def h_flip(image):
#   if np.random.rand(1) < 0.5:
#     return cp.flip(image, axis=-2)
#   return image

# def temporal_jitter(image, max_shift=3):
#   dt = np.random.randint(-max_shift, high=max_shift+1)
#   temp = cp.zeros_like(image)
#   if    dt < 0: temp[:,:,:,:dt] = image[:,:,:,-dt:]
#   elif  dt > 0: temp[:,:,:,dt:] = image[:,:,:,:-dt]
#   else: return image
#   return temp

# def spatial_jitter(image, max_shift=10):
#   dh = np.random.randint(-max_shift, high=max_shift+1)
#   dw = np.random.randint(-max_shift, high=max_shift+1)
  
#   _, H, W, _ = image.shape
#   temp = cp.zeros_like(image)

#   def idxs(shift, max_idx):
#     if shift >= 1: return  (shift, max_idx), (0, max_idx-shift)
#     elif shift==0: return (0,max_idx),  (0,max_idx)
#     else: return      (0,max_idx+shift),(-shift, max_idx)

#   (ihl, ihr), (thl, thr) = idxs(dh, H)
#   (iwl, iwr), (twl, twr)  = idxs(dw, W)
  
#   temp[:,thl:thr,twl:twr,:] = image[:,ihl:ihr,iwl:iwr,:]
#   return temp



class Wrapper(Dataset):
  def __init__(self, fs, transform=lambda x: x): 
    self.files = fs
    self.augment = transform
  def __len__(self): return len(self.files)  
  def __getitem__(self, idx):
    with np.load(self.files[idx]) as data:
      x = data['x']
      y = data['y']
    
    return self.augment(x),y


def Cifar10DVS(path, transform=lambda x: x, split=0.8):

  # Get Files #
  folders = os.listdir(path)
  files = np.empty(shape=0)
  for f in folders:
    f_path = os.path.join(path, f)
    fns = os.listdir(f_path)
    files = np.concatenate((files, np.array([os.path.join(f_path, fn) for fn in fns])), axis=0)

  # Shuffle #
  np.random.seed(0)
  np.random.shuffle(files)

  train = files[:int(len(files)*split)]
  test = files[int(len(files)*split):]


  return Wrapper(train, transform=transform), Wrapper(test)