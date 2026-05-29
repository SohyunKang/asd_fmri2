import torch
import scipy.io as sio

data = torch.load("/home/sohyunkang/asd_fmri2/experiments/w0_hitranet_20260310_172609_092015/fold1_best_aux.pt")

# tensor라면
if isinstance(data, torch.Tensor):
    data = data.numpy()

sio.savemat('aux.mat', {'data': data})