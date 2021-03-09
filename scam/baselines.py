from captum.attr import IntegratedGradients, Saliency, DeepLift, NoiseTunnel, GuidedGradCam
import torch
import numpy as np
import os

from scam.utils import save_image, normalize_image, image_to_tensor
from networks import init_network

torch.manual_seed(123)
np.random.seed(123)

def get_baselines(real_img, fake_img, real_class, fake_class, net_module, checkpoint_path, input_shape, channels, out_dir=None):
    imgs = [image_to_tensor(normalize_image(real_img).astype(np.float32)), image_to_tensor(normalize_image(fake_img).astype(np.float32))]
    classes = [real_class, fake_class]
    net = init_network(checkpoint_path, input_shape, net_module, channels, eval_net=True, require_grad=False)

    baseline = image_to_tensor(np.zeros(input_shape, dtype=np.float32))

    # IG
    net.zero_grad()
    ig = IntegratedGradients(net)
    ig_real, delta_real = ig.attribute(imgs[0], baseline, target=classes[0], return_convergence_delta=True)
    ig_fake, delta_fake = ig.attribute(imgs[1], baseline, target=classes[1], return_convergence_delta=True)

    # IG SMOOTH
    net.zero_grad()
    ig = IntegratedGradients(net)
    nt = NoiseTunnel(ig)
    igs_real = nt.attribute(imgs[0], baselines=baseline, target=classes[0], nt_type='smoothgrad_sq', n_samples=100, stdevs=0.2)
    igs_fake = nt.attribute(imgs[1], baselines=baseline, target=classes[1], nt_type='smoothgrad_sq', n_samples=100, stdevs=0.2)

    # SALIENCY
    net.zero_grad()
    saliency = Saliency(net)
    grads_real = saliency.attribute(imgs[0], 
                                    target=classes[0]) 
    grads_fake = saliency.attribute(imgs[1], 
                                    target=classes[1]) 

    # GUIDED GRAD CAM
    net.zero_grad()
    last_conv = [module for module in net.modules() if type(module) == torch.nn.Conv2d][-1]
    guided_gc = GuidedGradCam(net, last_conv)
    ggc_real = guided_gc.attribute(imgs[0], target=classes[0])
    ggc_fake = guided_gc.attribute(imgs[1], target=classes[1])

    attr = [ig_real.detach().numpy(), ig_fake.detach().numpy(), 
            igs_real.detach().numpy(), igs_fake.detach().numpy(),
            grads_real.detach().numpy(), grads_fake.detach().numpy(),
            ggc_real.detach().numpy(), ggc_fake.detach().numpy()]
    attr_names = ["ig_real", "ig_fake", "igsmooth_real", "igsmooth_fake", "grads_real", "grads_fake", "ggc_real", "ggc_fake"]

    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for att, name in zip(attr, attr_names):
            save_image(att[0,0,:,:], os.path.join(out_dir, name+".png"), renorm=False, norm=True)
