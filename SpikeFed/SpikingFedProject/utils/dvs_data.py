import numpy as np
from torch.utils.data import Dataset

import numpy as np
from torch.utils.data import Dataset
# import cupy as cp


# def c_flip(image):
#   temp = cp.zeros_like(image)
#   temp[0,:,:,:] = image[1,:,:,:]
#   temp[1,:,:,:] = image[0,:,:,:]
#   return temp
  
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
  def __init__(self,x,y, transform=None):
    super(Wrapper, self).__init__()
    self.images = x
    self.labels = y
    self.augment = transform
  def __len__(self): return self.images.shape[0]
  def __getitem__(self,idx): 
    image = self.images[idx]
    label = self.labels[idx]
    if self.augment is not None:
      image = self.augment(image)
    return image.astype(np.float32), label

CHANNEL_WARNING_PRINTED = False
def DvsGesture(path, num_users=5, transform=None):
    with np.load(path) as data:
        images = data['x']
        labels = data['y'].astype(int)
    print(f"Loaded DVS Gesture dataset. Shape: {images.shape}")  # Debugging

    # Ensure the shape is (B, C, H, W, T)
    if images.ndim == 4:  
        images = np.expand_dims(images, axis=1) 

    # Ensure C=2
    if images.shape[1] == 1:  
        if not CHANNEL_WARNING_PRINTED:
            print(f"⚠️ Warning: Expected input channel=2, but got 1. Adjusting... (This warning will only appear once.)")
            CHANNEL_WARNING_PRINTED = True  # Set flag to True
        images = np.repeat(images, 2, axis=1)  

    dataset = Wrapper(images, labels, transform=transform)

    # Split dataset for federated learning
    def dataset_iid(dataset, num_users):
        num_items = int(len(dataset) / num_users)
        dict_users, all_idxs = {}, list(range(len(dataset)))
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        return dict_users

    dict_users = dataset_iid(dataset, num_users)
    return dataset, dict_users
