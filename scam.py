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

    mrf_scores = []
    mask_sizes = []

    # Fixed for now:
    channels = 1
    attrs, attrs_names = get_attribution(real_img, fake_img,
                                         args.realclass, args.fakeclass,
                                         args.net, args.checkpoint,
                                         input_shape, channels, methods)


    for attr, name in zip(attrs, attrs_names):
        attr_imgs, attr_img_names, mrf_score, thr, mask_size, out_real, out_fake = get_mask(attr, real_img, fake_img, 
                                                                                            args.realclass, args.fakeclass, 
                                                                                            args.net, args.checkpoint, 
                                                                                            input_shape, channels)

        mrf_scores.append(mrf_score)
        mask_sizes.append(mask_size)

        base_dir = os.path.join(args.out, f"{name}")
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        for im, im_name in zip(attr_imgs, attr_img_names):
            save_image(im, os.path.join(base_dir, im_name + ".png"))

    for j in range(len(mrf_scores)):
        print(attrs_names[j], 
              mrf_scores[j].detach().cpu().numpy(), 
              mask_sizes[j], 
              mrf_scores[j].detach().cpu().numpy()/mask_sizes[j])
                
