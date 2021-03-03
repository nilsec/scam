import collections
import numpy as np
import cv2
import copy

from scam.gradients import get_gradients_from_layer
from scam.activations import get_activation_dict, get_layer_activations, project_layer_activations_to_input
from scam.utils import normalize_image
from networks import run_inference

def get_scammed(real_img, fake_img, real_class, fake_class, net, input_shape, layer_number, layer_prefix="features"):
    imgs = [real_img, fake_img]
    classes = [real_class, fake_class]
    
    grads = []
    for x,y in zip(imgs,classes):
        grads.append(get_gradients_from_layer(net, x, y, layer_number))

    acts_real = collections.defaultdict(list)
    acts_fake = collections.defaultdict(list)
    acts_real, out_real = get_activation_dict(net, [imgs[0]], acts_real)
    acts_fake, out_fake = get_activation_dict(net, [imgs[1]], acts_fake)
    acts = [acts_real, acts_fake]
    outs = [out_real, out_fake]
    
    layer_acts = []
    for act in acts:
        layer_acts.append(get_layer_activations(act, layer_number, layer_prefix))

    delta = grads[1] * (layer_acts[0] - layer_acts[1])
    delta_projected = project_layer_activations_to_input(net, input_shape, delta, layer_number)[0,:,:,:]
    
    channels = np.shape(delta_projected)[0]
    scam = np.zeros(np.shape(delta_projected)[1:])

    for c in range(channels):
        scam += delta_projected[c,:,:]

    scam /= np.max(np.abs(scam))
    return scam

def get_mask(attribution, real_img, fake_img, real_class, fake_class, net):
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


        print(np.shape(mask), np.shape(copyfrom), np.shape(copyto))
        mask_weight = cv2.GaussianBlur(mask_cp.astype(np.float), (11,11),0)
        copyto = np.array((copyto * (1 - mask_weight)) + (copyfrom * mask_weight), dtype=np.uint8)
        copied_canvas += np.array(mask_weight*copyfrom)
        copied_canvas_to = np.zeros(np.shape(copyfrom))
        copied_canvas_to+= np.array(mask_weight*copyto_ref)
        diff_copied = copied_canvas - copied_canvas_to

        imgs = [real_img, fake_img, copyto, copied_canvas,copied_canvas_to,diff_copied]
        
        fake_img_norm = normalize_image(copy.deepcopy(fake_img))
        out_fake = run_inference(net, fake_img_norm)
        real_img_norm = normalize_image(copy.deepcopy(real_img))
        out_real = run_inference(net, real_img_norm)
        im_copied_norm = normalize_image(copy.deepcopy(copyto))
        out_copyto = run_inference(net, im_copied_norm)

        print(out_real, out_fake)
        
        mrf_score = out_copyto[0][real_class] - out_fake[0][real_class]     
        
        thr -= 0.01

    return imgs, mrf_score, thr
