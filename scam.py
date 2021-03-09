import argparse
import os

from scam.utils import open_image
from scam import get_scammed, get_mask
from scam import get_baselines

parser = argparse.ArgumentParser()

parser.add_argument("--net", help="Name of network module in networks", required=True)
parser.add_argument("--checkpoint", help="Network checkpoint path", required=True)
parser.add_argument("--layer", help="Layer name", required=False, default=None)
parser.add_argument("--shape", help="Spatial image input shape", required=True, type=int)
parser.add_argument("--realimg", help="Path to real input image", required=True)
parser.add_argument("--fakeimg", help="Path to fake input image", required=True)
parser.add_argument("--realclass", help="Real class index", required=True, type=int)
parser.add_argument("--fakeclass", help="Fake class index", required=True, type=int)
parser.add_argument("--out", help="Output directory", required=False, default="scam_out")
parser.add_argument("--baselines", help="Calculate attribution baselines", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    input_shape = (args.shape, args.shape)
    real_img = open_image(args.realimg, flatten=True, normalize=False)
    fake_img = open_image(args.fakeimg, flatten=True, normalize=False)

    # Fixed for now:
    channels = 1
    scam = get_scammed(real_img, fake_img, args.realclass, 
                       args.fakeclass, args.net, args.checkpoint, 
                       input_shape, channels, args.layer)

    imgs, mrf_score, thr, out_real, out_fake = get_mask(scam, real_img, fake_img, 
                                                        args.realclass, args.fakeclass, args.net, 
                                                        args.checkpoint, input_shape, channels, args.out)

    if args.baselines:
        baseline_attributions = get_baselines(real_img, fake_img, args.realclass, 
                                              args.fakeclass, args.net, args.checkpoint, 
                                              input_shape, channels, os.path.join(args.out, "baselines"))

    print(f"SCAMed: Mask explains {float(mrf_score)} of feature difference.")
    print(f"{args.net}(real): {out_real.numpy()[0]}")
    print(f"{args.net}(fake): {out_fake.numpy()[0]}")
    print(f"See {args.out} for attribution.")
