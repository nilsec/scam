from captum.attr import IntegratedGradients, Saliency, DeepLift, NoiseTunnel, GuidedGradCam
import torch
import numpy as np
import os

from scam.utils import save_image, normalize_image, image_to_tensor
from networks import init_network

torch.manual_seed(123)
np.random.seed(123)

def get_baselines(real_img, fake_img, real_class, fake_class, net_module, checkpoint_path, input_shape, channels):
    imgs = [image_to_tensor(normalize_image(real_img).astype(np.float32)), 
            image_to_tensor(normalize_image(fake_img).astype(np.float32))]

    classes = [real_class, fake_class]
    net = init_network(checkpoint_path, input_shape, net_module, channels, eval_net=True, require_grad=False)

    
    # IG
    baseline = image_to_tensor(np.zeros(input_shape, dtype=np.float32))
    net.zero_grad()
    ig = IntegratedGradients(net)
    ig_real, delta_real = ig.attribute(imgs[0], baseline, target=classes[0], return_convergence_delta=True)
    ig_fake, delta_fake = ig.attribute(imgs[1], baseline, target=classes[1], return_convergence_delta=True)

    # IG DIFF
    ig_diff = ig_real - ig_fake

    # GRAD
    net.zero_grad()
    saliency = Saliency(net)
    grads_real = saliency.attribute(imgs[0], 
                                    target=classes[0]) 
    grads_fake = saliency.attribute(imgs[1], 
                                    target=classes[1]) 

    # GRAD DIFF
    grads_diff = grads_real - grads_fake

    # GC
    net.zero_grad()
    last_conv_layer = [(name,module) for name, module in net.named_modules() if type(module) == torch.nn.Conv2d][-1]
    layer_name = last_conv_layer[0]
    layer = last_conv_layer[1]
    layer_gc = LayerGradCam(net, layer)
    gc_real = layer_gc.attribute(imgs[0], target=classes[0])
    gc_fake = layer_gc.attribute(imgs[1], target=classes[1])

    # GC DIFF
    gc_diff = gc_real - gc_fake
    
    # GGC
    net.zero_grad()
    last_conv = [module for module in net.modules() if type(module) == torch.nn.Conv2d][-1]
    guided_gc = GuidedGradCam(net, last_conv)
    ggc_real = guided_gc.attribute(imgs[0], target=classes[0])
    ggc_fake = guided_gc.attribute(imgs[1], target=classes[1])

    # GGC DIFF
    ggc_diff = ggc_real - ggc_fake

    # DL
    net.zero_grad()
    dl = DeepLift(net)
    dl_real = dl.attribute(imgs[0], target=classes[0])
    dl_fake = dl.attribute(imgs[1], target=classes[1])

    # DL DIFF
    dl_diff = dl_real - dl_fake

    # INGRAD
    net.zero_grad()
    input_x_gradient = InputXGradient(net)
    ingrad_real = input_x_gradient.attribute(imgs[0], target=classes[0])
    ingrad_fake = input_x_gradient.attribute(imgs[1], target=classes[1])

    # INGRAD DIFF
    ingrad_diff = ingrad_real - ingrad_fake

    attr = [ig_real, 
            ig_fake,
            ig_diff,
            grads_real, 
            grads_fake,
            grads_diff,
            gc_real,
            gc_fake,
            gc_diff,
            ggc_real, 
            ggc_fake,
            ggc_diff,
            dl_real,
            dl_fake,
            dl_diff,
            ingrad_real,
            ingrad_fake,
            ingrad_diff]

    attr_names = ["ig_real", 
                  "ig_fake",
                  "ig_diff",
                  "grads_real", 
                  "grads_fake",
                  "grads_diff",
                  "gc_real",
                  "gc_fake",
                  "gc_diff",
                  "ggc_real", 
                  "ggc_fake",
                  "ggc_diff",
                  "dl_real",
                  "dl_fake",
                  "dl_diff",
                  "ingrad_real",
                  "ingrad_fake",
                  "ingrad_diff"]

    attr = [a.detach().cpu().numpy() for a in attr]
    attr_norm = [a[0,0,:,:]/np.max(np.abs(a[0,0,:,:])) for a in attr]

    return attr_norm, attr_names
