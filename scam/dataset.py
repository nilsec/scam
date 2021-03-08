from shutil import copy
import os

from scam.utils import open_image
from networks import init_network, run_inference

def create_dataset(data_dir, image_pairs, real_class, fake_class, net_module, checkpoint_path, input_shape, input_nc, real_thr=0.8, fake_thr=0.8):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    net = init_network(checkpoint_path, input_shape, net_module, input_nc, eval_net=True, require_grad=False)
    dataset = []
    
    idx = 0 
    for img_pair in image_pairs:
        real = open_image(img_pair[0])
        fake = open_image(img_pair[1])
        
        out_real = run_inference(net, real).numpy()[0][real_class]
        out_fake = run_inference(net, fake).numpy()[0][fake_class]

        if out_real > real_thr and out_fake > fake_thr:
            copy(img_pair[0], data_dir + f"/real_{idx}.png")
            copy(img_pair[1], data_dir + f"/fake_{idx}.png")

        idx += 1
