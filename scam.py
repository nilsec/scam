from scam.utils import open_image
from scam import get_scammed, get_mask

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--net", help="Name of network module in networks", required=True)
parser.add_argument("--checkpoint", help="Network checkpoint path", required=True)
parser.add_argument("--layer", help="Layer number", required=True, type=int)
parser.add_argument("--shape", help="Spatial image input shape", required=True, type=int)
parser.add_argument("--realimg", help="Path to real input image", required=True)
parser.add_argument("--fakeimg", help="Path to fake input image", required=True)
parser.add_argument("--realclass", help="Real class index", required=True, type=int)
parser.add_argument("--fakeclass", help="Fake class index", required=True, type=int)
parser.add_argument("--out", help="Output directory", required=False, default="scam_out")

if __name__ == "__main__":
    args = parser.parse_args()
    input_shape = (args.shape, args.shape)
    real_img = open_image(args.realimg, flatten=True, normalize=False)
    fake_img = open_image(args.fakeimg, flatten=True, normalize=False)
    # Fixed for now:
    channels = 1
    scam = get_scammed(real_img, fake_img, args.realclass, args.fakeclass, args.net, args.checkpoint, input_shape, channels, args.layer)
    imgs, mrf_score, thr = get_mask(scam, real_img, fake_img, args.realclass, args.fakeclass, args.net, args.checkpoint, input_shape, channels, args.out)
    print(f"SCAMed: Mask explains {mrf_score[0]} of feature difference. See {args.out} for attribution.")
