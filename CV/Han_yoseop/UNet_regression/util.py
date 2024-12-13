import os

import numpy as np
import torch

from scipy.stats import poisson
from skimage.transform import rescale, resize

# 모델 save, load, 추후 output file 관련 시각화 등 utility 기능 구현
# 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    torch.save({'net' : net.state_dict(), 'optim': optim.state_dict()},
               "./%s/model_epoch%d.pth" % (ckpt_dir,epoch))

# 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f : int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1])) # 가장 최신 버전의 가중치 가져오기

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

# artifact 적용하는 util code

# sampling code (적용할 이미지, 적용할 샘플링 기법)
def add_sampling(img, type="random", opts=None):
    sz = img.shape

    if type == "uniform":
        ds_y = opts[0].astype(np.int32)
        ds_x = opts[1].type(np.int32)

        msk = np.zeros(sz)
        msk[::ds_y,::ds_x, :] = 1 # 돌아가는 원리?

        dst = img * msk
    elif type == "random":
        prob = opts[0]
        rnd = np.random.rand(sz[0], sz[1], sz[2])
        msk = (rnd > prob).astype(np.float32)

        dst = img * msk
    elif type == "gaussian":
        x0 = opts[0]
        y0 = opts[1]
        sgmx = opts[2]
        sgmy = opts[3]
        a = opts[4]

        ly = np.linspace(-1, 1, sz[0])
        lx = np.linspace(-1, 1, sz[1])

        x, y = np.meshgrid(lx, ly)

        gaus = a * np.exp(-((x - x0)**2/(2*sgmx**2) + (y - y0)**2/(2*sgmy**2)))
        gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, sz[2]))
        rnd = np.random.rand(sz[0], sz[1], sz[2])
        msk = (rnd < gaus).astype(np.float32)

        dst = img * msk

    return dst

# noise 부여하기
def add_noise(img, type="random", opts=None):
    sz = img.shape

    if type == "random":
        sgm = opts[0] # sigma
        noise = sgm/255.0 * np.random.randn(sz[0], sz[1], sz[2])
        dst = img + noise
    elif type == "poisson":
        dst = poisson.rvs(255.0 * img) / 255.0
        noise = dst - img

    return dst

# re-scailing 부여 (blurring)
def add_blur(img, type="bilinear", opts=None):
    if type == "nearest":
        order = 0
    elif type == "bilinear":
        order = 1
    elif type == "biquadratic":
        order = 2
    elif type == "bicubic":
        order = 3
    elif type == "biquartic":
        order = 4
    elif type == "biquintic":
        order = 5

    sz = img.shape

    dw = opts[0] # downsampling 비율

    if len(opts) == 1:
        # opts에 2번째 인자가 없다면
        keepdim = True # dim을 input 영상의 dim에 맞게끔 다시 upsample
    else:
        keepdim = opts[1]

    # rescale vs resize 비교
    # rescale은 직접 downsampling 비율을 지정해줘야 함 dw
    # reshape는 output shape만 지정해주면 알아서 출력 크기에 맞춤
    # dst = rescale(img, scale=(dw, dw, 1), order=order)
    dst = resize(img, output_shape=(sz[0] // dw, sz[1] // dw, sz[2]), order=order)

    if keepdim:
        dst = resize(dst, output_shape=(sz[0], sz[1], sz[2]), order=order)

    return dst
