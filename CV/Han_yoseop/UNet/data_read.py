# 기존 data는 512 x 512 의 30 frame으로 구성되어 있음
# 이것을 30 frame이 아닌 1 frame씩 받아오도록 전처리

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 데이터 불러오기
dir_data = './datasets'

name_label = 'train-labels.tif'
name_input = 'train-volume.tif'

img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

ny, nx = img_label.size # x,y 개수라 그냥 이렇게 한 듯 >> 512
nframe = img_label.n_frames # 30

# 30개의 프레임을 학습, 검증, 평가 용으로 분리
nframe_train = 24
nframe_val = 3
nframe_test = 3

dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test= os.path.join(dir_data, 'test')

if not os.path.exists(dir_save_train):
    os.mkdir(dir_save_train)

if not os.path.exists(dir_save_val):
    os.mkdir(dir_save_val)

if not os.path.exists(dir_save_test):
    os.mkdir(dir_save_test)

# 각 디렉토리에 frame들 random하게 저장
id_frame = np.arange(nframe) # 30개를 나열 0 ~ 29의 numpy 생성
np.random.shuffle(id_frame) # 0 ~ 29의 numpy인 id_frame을 셔플해서 다시 저장

# train 데이터셋을 저장하는 구문
offset_nframe = 0

for i in range(nframe_train):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

# val 데이터셋을 저장하는 구문
offset_nframe += nframe_train

for i in range(nframe_val):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

# test 데이터셋을 저장하는 구문
offset_nframe += nframe_val

for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

# 이렇게 분리된 dataset matplotlib를 통해 출력
plt.subplot(121)
plt.imshow(label_, cmap='gray')
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('input')

plt.show()