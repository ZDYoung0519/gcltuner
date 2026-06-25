import torch    
import argparse
from mmengine import print_log
from xtuner.model.utils import guess_load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='merge checkpoints')
    parser.add_argument('--cur-ckpt', help='the dir to save logs and models')
    parser.add_argument('--pre-ckpt', help='the dir to save logs and models')
    parser.add_argument('--output', help='the dir to save logs and models')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    cur_ckpt = guess_load_checkpoint(args.cur_ckpt)
    pre_ckpt = guess_load_checkpoint(args.pre_ckpt)

    print("Before merge:")
    print("Current ckpt", cur_ckpt.keys())
    print("Previous ckpt", pre_ckpt.keys())

    for k, v in cur_ckpt.items():
        if k not in pre_ckpt.keys():
            pre_ckpt[k] = v
            print_log(f"{k} is updated in {args.pre_ckpt}")
        else:
            pre_ckpt[k] = v
            print_log(f"Warining: {k} already exisits in {args.pre_ckpt}")
    
    print("After merge:")
    print("Previous ckpt", pre_ckpt.keys())

    torch.save(pre_ckpt, args.output)
    print_log(f"Saved at {args.output}")

if __name__ == '__main__':
    main()
