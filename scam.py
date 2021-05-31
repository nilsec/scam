import argparse
import os

from scam.utils import open_image, save_image
from scam import get_attribution, get_mask

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
parser.add_argument("--ig", help="Turn OFF IG attr", action="store_false")
parser.add_argument("--grads", help="Turn OFF grads attr", action="store_false")
parser.add_argument("--gc", help="Turn OFF GC attr", action="store_false")
parser.add_argument("--ggc", help="Turn OFF GGC attr", action="store_false")
parser.add_argument("--dl", help="Turn OFF DL attr", action="store_false")
parser.add_argument("--ingrad", help="Turn OFF ingrad attr", action="store_false")
parser.add_argument("--random", help="Turn OFF random attr", action="store_false")


if __name__ == "__main__":
    args = parser.parse_args()
    input_shape = (args.shape, args.shape)
    real_img = open_image(args.realimg, flatten=True, normalize=False)
    fake_img = open_image(args.fakeimg, flatten=True, normalize=False)

    methods = []

    if args.ig:
        methods.append("ig")
    if args.grads:
        methods.append("grads")
    if args.gc:
        methods.append("gc")
    if args.ggc:
        methods.append("ggc")
    if args.dl:
        methods.append("dl")
    if args.ingrad:
        methods.append("ingrad")
    if args.random:
        methods.append("random")

    mrf_scores = []
    mask_sizes = []

    # Fixed for now:
    channels = 1
    attrs, attrs_names = get_attribution(real_img, fake_img,
                                         args.realclass, args.fakeclass,
                                         args.net, args.checkpoint,
                                         input_shape, channels, methods)


    for attr, name in zip(attrs, attrs_names):
        result_dict, img_names, imgs_all, img_thresholds = get_mask(attr, real_img, fake_img, 
                                                                    args.realclass, args.fakeclass, 
                                                                    args.net, args.checkpoint, 
                                                                    input_shape, channels)

        method_dir = os.path.join(args.out, f"{name}")
        if not os.path.exists(method_dir):
            os.makedirs(method_dir)

        for mask_imgs, thr in zip(imgs_all, img_thresholds):
            threshold_dir = os.path.join(method_dir, f"t_{thr}")
            if not os.path.exists(threshold_dir):
                os.makedirs(threshold_dir)
            for mask_im, mask_name in zip(mask_imgs, img_names):
                save_image(mask_im, os.path.join(threshold_dir, mask_name + ".png"))

        with open(os.path.join(method_dir, "results.txt"), 'w+') as f:
            print(result_dict, file=f)
