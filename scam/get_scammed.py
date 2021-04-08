import collections
import numpy as np
import os
import cv2
import copy
import torch

from scam.gradients import get_gradients_from_layer
from scam.activations import get_activation_dict, get_layer_activations, project_layer_activations_to_input
from scam.utils import normalize_image, save_image
from networks import run_inference, init_network

def get_scammed(real_img, fake_img, real_class, fake_class, net_module, checkpoint_path, input_shape, input_nc, layer_name=None):
    """
        real_img: Unnormalized (0-255) 2D image

        fake_img: Unnormalized (0-255) 2D image

        *_class: Index of real and fake class corresponding to network output

        net_module: Name of file and class name of the network to use. Must be placed in networks subdirectory

        checkpoint_path: Checkpoint of network.

        input_shape: Spatial input shape of network

        input_nc: Number of input channels.

        layer_name: Name of the conv layer to use (defaults to last)

    """


    if len(np.shape(fake_img)) != len(np.shape(real_img)) !=2:
        raise ValueError("Input images need to be two dimensional")
    
    imgs = [normalize_image(real_img), normalize_image(fake_img)]
    classes = [real_class, fake_class]

    if layer_name is None:
        net = init_network(checkpoint_path, input_shape, net_module, input_nc, eval_net=True, require_grad=False)
        last_conv_layer = [(name,module) for name, module in net.named_modules() if type(module) == torch.nn.Conv2d][-1]
        layer_name = last_conv_layer[0]
        layer = last_conv_layer[1]
   
    grads = []
    for x,y in zip(imgs,classes):
        grad_net = init_network(checkpoint_path, input_shape, net_module, input_nc, eval_net=True, require_grad=False)
        grads.append(get_gradients_from_layer(grad_net, x, y, layer_name))

    acts_real = collections.defaultdict(list)
    acts_fake = collections.defaultdict(list)

    activation_net = init_network(checkpoint_path, input_shape, net_module, input_nc, eval_net=True, require_grad=False)
    acts_real, out_real = get_activation_dict(activation_net, [imgs[0]], acts_real)

    activation_net = init_network(checkpoint_path, input_shape, net_module, input_nc, eval_net=True, require_grad=False)
    acts_fake, out_fake = get_activation_dict(activation_net, [imgs[1]], acts_fake)

    acts = [acts_real, acts_fake]
    outs = [out_real, out_fake]
    
    layer_acts = []
    for act in acts:
        layer_acts.append(get_layer_activations(act, layer_name))

    net = init_network(checkpoint_path, input_shape, net_module, input_nc, eval_net=True, require_grad=False)
    delta_fake = grads[1] * (layer_acts[0] - layer_acts[1])
    delta_real = grads[0] * (layer_acts[1] - layer_acts[0])
    delta_fake_projected = project_layer_activations_to_input(net, (input_nc, input_shape[0], input_shape[1]), delta_fake, layer_name)[0,:,:,:]
    delta_real_projected = project_layer_activations_to_input(net, (input_nc, input_shape[0], input_shape[1]), delta_real, layer_name)[0,:,:,:]
    
    channels = np.shape(delta_fake_projected)[0]
    scam_0 = np.zeros(np.shape(delta_fake_projected)[1:])
    scam_1 = np.zeros(np.shape(delta_real_projected)[1:])

    for c in range(channels):
        scam_0 += delta_fake_projected[c,:,:]
        scam_1 += delta_real_projected[c,:,:]

    scam_0 /= np.max(np.abs(scam_0))
    scam_1 /= np.max(np.abs(scam_1))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.tensor(scam_0, device=device), torch.tensor(scam_1, device=device)

def get_mask(attribution, real_img, fake_img, real_class, fake_class, 
             net_module, checkpoint_path, input_shape, input_nc):
    """
    attribution: 2D array <= 1 indicating pixel importance
    """

    net = init_network(checkpoint_path, input_shape, net_module, input_nc, eval_net=True, require_grad=False)
    result_dict = {}
    img_names = ["attr", "real", "fake", "hybrid", "mask_real", "mask_fake", "mask_residual"]
    imgs_all = []
    img_thresholds = [0, 0.2, 0.4, 0.6, 0.8, 0.99]

    for k in range(0,100):
        thr = k * 0.01
        copyfrom = copy.deepcopy(real_img)
        copyto = copy.deepcopy(fake_img)
        copyto_ref = copy.deepcopy(fake_img)
        copied_canvas = np.zeros(np.shape(copyfrom))
        mask = np.array(attribution > thr, dtype=np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_size = np.sum(mask)
        mask_cp = copy.deepcopy(mask)

        mask_weight = cv2.GaussianBlur(mask_cp.astype(np.float), (11,11),0)
        copyto = np.array((copyto * (1 - mask_weight)) + (copyfrom * mask_weight), dtype=np.float)

        copied_canvas += np.array(mask_weight*copyfrom)
        copied_canvas_to = np.zeros(np.shape(copyfrom))
        copied_canvas_to += np.array(mask_weight*copyto_ref)
        diff_copied = copied_canvas - copied_canvas_to
        
        fake_img_norm = normalize_image(copy.deepcopy(fake_img))
        out_fake = run_inference(net, fake_img_norm)
        
        real_img_norm = normalize_image(copy.deepcopy(real_img))
        out_real = run_inference(net, real_img_norm)

        im_copied_norm = normalize_image(copy.deepcopy(copyto))
        out_copyto = run_inference(net, im_copied_norm)

        imgs = [attribution, real_img_norm, fake_img_norm, im_copied_norm, normalize_image(copied_canvas), 
                normalize_image(copied_canvas_to), normalize_image(diff_copied)]

        #if thr in img_thresholds:
        imgs_all.append(imgs)

        mrf_score = out_copyto[0][real_class] - out_fake[0][real_class]     
        result_dict[thr] = [float(mrf_score.detach().cpu().numpy()), mask_size]

    return result_dict, img_names, imgs_all, img_thresholds
