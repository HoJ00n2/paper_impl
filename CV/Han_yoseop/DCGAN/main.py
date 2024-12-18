import argparse

from train import *

# Parser object 호출(초기화)
parser = argparse.ArgumentParser(description="train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Parser에 추가할 argument들 등록하기 (등록하면 이후 터미널을 통해 자동적으로 argument들을 입력으로 넣을 수 있음)
parser.add_argument("--lr", default=2e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=128, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=1, type=int, dest="num_epoch")

# Parser에 추가할 config 설정들 등록
parser.add_argument("--data_dir", default="../dataset/img_align_celeba/", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--task", default="DCGAN", choices=["DCGAN"], type=str, dest="task")

# train or test mode 설정
parser.add_argument("--mode", default="train", type=str, dest="mode")
# 학습 처음부터 or 이어할지
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")
# opts 설정 nargs를 통해 유동적인 인자 개수 받기 (opts는 어떤 기법 적용하냐에 따라 2 ~ 5개의 값을 가짐)

# SRResNet이 아닐 경우
parser.add_argument("--opts", nargs='+' ,default=["random", 4], dest="opts")
# 2번째 인자는 keepdim에 대한 것 0으로하면 dim 유지를 안함으로써 input, output resolution을 다르게 만듦 -> SRResNet일 경우
# parser.add_argument("--opts", nargs='+' ,default=["random", 4, 0], dest="opts")

# 내가 조절하고픈 사이즈 지정 (nx, ny) for resize
parser.add_argument("--ny", default=64, type=int, dest="ny")
parser.add_argument("--nx", default=64, type=int, dest="nx")
parser.add_argument("--nch", default=3, type=int, dest="nch")
parser.add_argument("--nker", default=128, type=int, dest="nker")
parser.add_argument("--nblk", default=16, type=int, dest="nblk")

# 학습 네트워크 선택 (향후 추가될 수 있으므로 []로 관리)
parser.add_argument("--network", default="DCGAN", choices=["unet", "resnet", "srresnet", "DCGAN"], type=str, dest="network")

parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")

# parser에 등록한 argument들 사용(파싱)
args = parser.parse_args() # args에는 각 parser의 arguemnt들이 들어있음

if __name__ == "__main__":
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)