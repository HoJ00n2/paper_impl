import os

import torch

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