import torch
import argparse
import numpy as np
from models.TMC import ETMC

def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=128)
    parser.add_argument("--train_data_path", type=str, default="datasets/train/fakeav*")
    parser.add_argument("--val_data_path", type=str, default="datasets/val/fakeav*")
    parser.add_argument("--LOAD_SIZE", type=int, default=256)
    parser.add_argument("--FINE_SIZE", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=512)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.3)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--name", type=str, default="ReleasedVersion")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--savedir", type=str, default="./savepath/ETMC/nyud2/")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--annealing_epoch", type=int, default=10)
    parser.add_argument("--nb_samp", type=int, default=64600)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--blocks", nargs="*", type=int, default=[2, 4])
    parser.add_argument("--nb_fc_node", type=int, default=1024)
    parser.add_argument("--gru_node", type=int, default=1024)
    parser.add_argument("--nb_gru_layer", type=int, default=3)
    parser.add_argument("--device", type=str, default='cpu')

parser = argparse.ArgumentParser(description="Train Models")
get_args(parser)
args, remaining_args = parser.parse_known_args()

multimodal_model = ETMC(args)


video = torch.randn((1, 3, 256, 256))

import librosa
x, sr = librosa.load('my_result.wav')
print(x.shape)

spec_x = torch.unsqueeze(torch.Tensor(x), dim = 0)
print(spec_x.shape)

out = multimodal_model(video, spec_x)

print(out)


