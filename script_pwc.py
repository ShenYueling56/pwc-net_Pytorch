import sys
import cv2
import torch
import numpy as np
from math import ceil
from torch.autograd import Variable
from scipy.ndimage import imread
import models
"""
Contact: Deqing Sun (deqings@nvidia.com); Zhile Ren (jrenzhile@gmail.com)
"""
def writeFlowFile(filename,uv):
	"""
	According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein  
	Contact: dqsun@cs.brown.edu
	Contact: schar@middlebury.edu
	"""
	TAG_STRING = np.array(202021.25, dtype=np.float32)
	if uv.shape[2] != 2:
		sys.exit("writeFlowFile: flow must have two bands!");
	H = np.array(uv.shape[0], dtype=np.int32)
	W = np.array(uv.shape[1], dtype=np.int32)
	with open(filename, 'wb') as f:
		f.write(TAG_STRING.tobytes())
		f.write(W.tobytes())
		f.write(H.tobytes())
		f.write(uv.tobytes())

def flow2rgb(flow_map_np, max_value=None):
    _, h, w = flow_map_np.shape
    hsv = np.ones((h,w,3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow_map_np[0].squeeze(), flow_map_np[1].squeeze())
    hsv[..., 0] = ang / (2 * np.pi) * 180
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

im1_fn = 'data/input1_1.jpg';
im2_fn = 'data/input1_2.jpg';
flow_fn = 'data/frame_0010.flo';

if len(sys.argv) > 1:
    im1_fn = sys.argv[1]
if len(sys.argv) > 2:
    im2_fn = sys.argv[2]
if len(sys.argv) > 3:
    flow_fn = sys.argv[3]

pwc_model_fn = './pwc_net_chairs.pth.tar';

im_all = [imread(img) for img in [im1_fn, im2_fn]]
im_all = [im[:, :, :3] for im in im_all]

# rescale the image size to be multiples of 64
divisor = 64.
H = im_all[0].shape[0]
W = im_all[0].shape[1]

H_ = int(ceil(H/divisor) * divisor)
W_ = int(ceil(W/divisor) * divisor)
for i in range(len(im_all)):
	im_all[i] = cv2.resize(im_all[i], (W_, H_))

for _i, _inputs in enumerate(im_all):
	im_all[_i] = im_all[_i][:, :, ::-1]
	im_all[_i] = 1.0 * im_all[_i]/255.0
	
	im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
	im_all[_i] = torch.from_numpy(im_all[_i])
	im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])	
	im_all[_i] = im_all[_i].float()
    
im_all = torch.autograd.Variable(torch.cat(im_all,1).cuda(), volatile=True)

net = models.pwc_dc_net(pwc_model_fn)
net = net.cuda()
net.eval()

flo = net(im_all)
flo = flo[0] * 20.0
flo = flo.cpu().data.numpy()

# # scale the flow back to the input size
# flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2) #
# u_ = cv2.resize(flo[:,:,0],(W,H))
# v_ = cv2.resize(flo[:,:,1],(W,H))
# u_ *= W/ float(W_)
# v_ *= H/ float(H_)
# flo = np.dstack((u_,v_))

# import ipdb; ipdb.set_trace()
# writeFlowFile(flow_fn, flo)
rgb_flow = flow2rgb(flo)
# Image.fromarray(rgb_flow, 'RGB').show()
cv2.imwrite('data/rgb_flow1_1.png', rgb_flow)