import collections
import numpy as np
import os
import cv2
import copy

from scam.gradients import get_gradients_from_layer
from scam.activations import get_activation_dict, get_layer_activations, project_layer_activations_to_input
from scam.utils import normalize_image, save_image
from networks import run_inference, init_network

def get_scammed(real_img, fake_img, real_class, fake_class, net_module, checkpoint_path, input_shape, input_nc, layer_number, layer_prefix="features"):
    """
        real_img: Unnormalized (0-255) 2D image

        fake_img: Unnormalized (0-255) 2D image

        *_class: Index of real and fake class corresponding to network output

        net_module: Name of file and class name of the network to use. Must be placed in networks subdirectory

        checkpoint_path: Checkpoint of network.

        input_shape: Spatial input shape of network

        input_nc: Number of input channels.

        layer_number: Number of the last convolutional layer

        layer_prefix: Name of the layer
    """


    if len(np.shape(fake_img)) != len(np.shape(real_img)) !=2:
        raise ValueError("Input images need to be two dimensional")
    
    imgs = [normalize_image(real_img), normalize_image(fake_img)]
    classes = [real_class, fake_class]
    
    grads = []
    for x,y in zip(imgs,classes):
        grad_net = init_network(checkpoint_path, input_shape, net_module, input_nc, eval_net=True, require_grad=False)
        grads.append(get_gradients_from_layer(grad_net, x, y, layer_number))

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
        layer_acts.append(get_layer_activations(act, layer_number, layer_prefix))

    net = init_network(checkpoint_path, input_shape, net_module, input_nc, eval_net=True, require_grad=False)
    delta = grads[1] * (layer_acts[0] - layer_acts[1])
    delta_projected = project_layer_activations_to_input(net, (input_nc, input_shape[0], input_shape[1]), delta, layer_number)[0,:,:,:]
    
    channels = np.shape(delta_projected)[0]
    scam = np.zeros(np.shape(delta_projected)[1:])

    for c in range(channels):
        scam += delta_projected[c,:,:]

    scam /= np.max(np.abs(scam))
    return scam

def get_mask(attribution, real_img, fake_img, real_class, fake_class, net_module, checkpoint_path, input_shape, input_nc, out_dir=None):
    """
    attribution: 2D array <= 1 indicating pixel importance
    """


    net = init_network(checkpoint_path, input_shape, net_module, input_nc, eval_net=True, require_grad=False)

    mrf_score = 0
    thr = 0.9
    while mrf_score<0.5 and thr>=0:
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
        copied_canvas_to+= np.array(mask_weight*copyto_ref)
        diff_copied = copied_canvas - copied_canvas_to

        
        fake_img_norm = normalize_image(copy.deepcopy(fake_img))
        out_fake = run_inference(net, fake_img_norm)
        
        real_img_norm = normalize_image(copy.deepcopy(real_img))
        out_real = run_inference(net, real_img_norm)

        im_copied_norm = normalize_image(copy.deepcopy(copyto))
        out_copyto = run_inference(net, im_copied_norm)

        imgs = [attribution, real_img_norm, fake_img_norm, im_copied_norm, normalize_image(copied_canvas), 
                normalize_image(copied_canvas_to), normalize_image(diff_copied)]

        mrf_score = out_copyto[0][real_class] - out_fake[0][real_class]     
        
        thr -= 0.01

    if out_dir is not None:
        img_names = ["attr","real", "fake", "hybrid", "mask_real", "mask_fake", "mask_residual"]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for im, im_name in zip(imgs,img_names):
            save_image(im, os.path.join(out_dir, im_name + ".png"))

    return imgs, mrf_score, thr
