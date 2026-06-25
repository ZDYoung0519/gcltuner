import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./pretrained_weights/mm_projector.bin')
    parser.add_argument('--dst_path', type=str, default='./pretrained_weights/mm_projector_xtuner.pt')
    args = parser.parse_args()

    projector_weight = torch.load(args.src_path, map_location=torch.device('cpu'), weights_only=False)

    projector_weight_ = {}
    for k, v in projector_weight.items():
        new_k = k.replace('model.mm_projector', 'projector.model')
        projector_weight_[new_k] = v
        print("{} -> {}, shape: {}".format(k, new_k, v.shape))
    torch.save(projector_weight_, args.dst_path)
    print('Done! Output saved to {}'.format(args.dst_path))


if __name__ == '__main__':
    main()


