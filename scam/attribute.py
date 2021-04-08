from captum.attr import IntegratedGradients, Saliency, DeepLift, NoiseTunnel, GuidedGradCam, InputXGradient, DeepLift, LayerGradCam, GuidedBackprop
import torch
import numpy as np
import os

from scam.utils import save_image, normalize_image, image_to_tensor
from scam.activations import project_layer_activations_to_input
from scam import get_scammed
from networks import init_network

torch.manual_seed(123)
np.random.seed(123)

def get_attribution(real_img, 
                    fake_img, 
                    real_class, 
                    fake_class, 
                    net_module, 
                    checkpoint_path, 
                    input_shape, 
                    channels,
                    methods=["ig", "grads", "gc", "ggc", "dl", "ingrad"]):


    imgs = [image_to_tensor(normalize_image(real_img).astype(np.float32)), 
            image_to_tensor(normalize_image(fake_img).astype(np.float32))]

    classes = [real_class, fake_class]
    net = init_network(checkpoint_path, input_shape, net_module, channels, eval_net=True, require_grad=False)

    attrs = []
    attrs_names = []

    if "gc" in methods:
        net.zero_grad()
        last_conv_layer = [(name,module) for name, module in net.named_modules() if type(module) == torch.nn.Conv2d][-1]
        layer_name = last_conv_layer[0]
        layer = last_conv_layer[1]
        layer_gc = LayerGradCam(net, layer)
        gc_real = layer_gc.attribute(imgs[0], target=classes[0])
        gc_fake = layer_gc.attribute(imgs[1], target=classes[1])


        # PROJECT
        last_conv_layer = [(name,module) for name, module in net.named_modules() if type(module) == torch.nn.Conv2d][-1]
        layer_name = last_conv_layer[0]
        layer = last_conv_layer[1]

        gc_real = project_layer_activations_to_input(net, (1, input_shape[0], input_shape[1]), gc_real.cpu().detach().numpy(), layer_name)
        gc_fake = project_layer_activations_to_input(net, (1, input_shape[0], input_shape[1]), gc_fake.cpu().detach().numpy(), layer_name)

        attrs.append(torch.tensor(gc_real[0,0,:,:]))
        attrs_names.append("gc_real")

        attrs.append(torch.tensor(gc_fake[0,0,:,:]))
        attrs_names.append("gc_fake")

        # SCAM
        gc_diff_0, gc_diff_1 = get_scammed(real_img, fake_img, real_class, 
                                     fake_class, net_module, checkpoint_path, 
                                     input_shape, channels, None)
        attrs.append(gc_diff_0)
        attrs_names.append("gc_diff_0")

        attrs.append(gc_diff_1)
        attrs_names.append("gc_diff_1")

    if "ggc" in methods:
        net.zero_grad()
        last_conv = [module for module in net.modules() if type(module) == torch.nn.Conv2d][-1]
        guided_gc = GuidedGradCam(net, last_conv)
        ggc_real = guided_gc.attribute(imgs[0], target=classes[0])
        ggc_fake = guided_gc.attribute(imgs[1], target=classes[1])

        attrs.append(ggc_real[0,0,:,:])
        attrs_names.append("ggc_real")

        attrs.append(ggc_fake[0,0,:,:])
        attrs_names.append("ggc_fake")

        net.zero_grad()
        gbp = GuidedBackprop(net)
        gbp_real = gbp.attribute(imgs[0], target=classes[0])
        gbp_fake = gbp.attribute(imgs[1], target=classes[1])
        
        attrs.append(gbp_real[0,0,:,:])
        attrs_names.append("gbp_real")

        attrs.append(gbp_fake[0,0,:,:])
        attrs_names.append("gbp_fake")

        ggc_diff_0 = gbp_real[0,0,:,:] * gc_diff_0
        ggc_diff_1 = gbp_fake[0,0,:,:] * gc_diff_1

        attrs.append(ggc_diff_0)
        attrs_names.append("ggc_diff_0")

        attrs.append(ggc_diff_1)
        attrs_names.append("ggc_diff_1")

    # IG
    if "ig" in methods:
        baseline = image_to_tensor(np.zeros(input_shape, dtype=np.float32))
        net.zero_grad()
        ig = IntegratedGradients(net)
        ig_real, delta_real = ig.attribute(imgs[0], baseline, target=classes[0], return_convergence_delta=True)
        ig_fake, delta_fake = ig.attribute(imgs[1], baseline, target=classes[1], return_convergence_delta=True)
        ig_diff_0, delta_diff = ig.attribute(imgs[0], imgs[1], target=classes[0], return_convergence_delta=True)
        ig_diff_1, delta_diff = ig.attribute(imgs[1], imgs[0], target=classes[1], return_convergence_delta=True)

        attrs.append(ig_real[0,0,:,:])
        attrs_names.append("ig_real")

        attrs.append(ig_fake[0,0,:,:])
        attrs_names.append("ig_fake")

        attrs.append(ig_diff_0[0,0,:,:])
        attrs_names.append("ig_diff_0")

        attrs.append(ig_diff_1[0,0,:,:])
        attrs_names.append("ig_diff_1")

        
    # DL
    if "dl" in methods:
        net.zero_grad()
        dl = DeepLift(net)
        dl_real = dl.attribute(imgs[0], target=classes[0])
        dl_fake = dl.attribute(imgs[1], target=classes[1])
        dl_diff_0 = dl.attribute(imgs[0], baselines=imgs[1], target=classes[0])
        dl_diff_1 = dl.attribute(imgs[1], baselines=imgs[0], target=classes[1])

        attrs.append(dl_real[0,0,:,:])
        attrs_names.append("dl_real")

        attrs.append(dl_fake[0,0,:,:])
        attrs_names.append("dl_fake")

        attrs.append(dl_diff_0[0,0,:,:])
        attrs_names.append("dl_diff_0")

        attrs.append(dl_diff_1[0,0,:,:])
        attrs_names.append("dl_diff_1")

    # INGRAD
    if "ingrad" in methods:
        net.zero_grad()
        saliency = Saliency(net)
        grads_real = saliency.attribute(imgs[0], 
                                        target=classes[0]) 
        grads_fake = saliency.attribute(imgs[1], 
                                        target=classes[1]) 

        attrs.append(grads_real[0,0,:,:])
        attrs_names.append("grads_real")

        attrs.append(grads_fake[0,0,:,:])
        attrs_names.append("grads_fake")

        net.zero_grad()
        input_x_gradient = InputXGradient(net)
        ingrad_real = input_x_gradient.attribute(imgs[0], target=classes[0])
        ingrad_fake = input_x_gradient.attribute(imgs[1], target=classes[1])

        ingrad_diff_0 = grads_fake * (imgs[0] - imgs[1])
        ingrad_diff_1 = grads_real * (imgs[1] - imgs[0])

        attrs.append(torch.abs(ingrad_real[0,0,:,:]))
        attrs_names.append("ingrad_real")

        attrs.append(torch.abs(ingrad_fake[0,0,:,:]))
        attrs_names.append("ingrad_fake")

        attrs.append(torch.abs(ingrad_diff_0[0,0,:,:]))
        attrs_names.append("ingrad_diff_0")

        attrs.append(torch.abs(ingrad_diff_1[0,0,:,:]))
        attrs_names.append("ingrad_diff_1")

    attrs = [a.detach().cpu().numpy() for a in attrs]
    attrs_norm = [a/np.max(np.abs(a)) for a in attrs]

    return attrs_norm, attrs_names
