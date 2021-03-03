import collections
import numpy as np

from scam.gradients import get_gradients_from_layer
from scam.activations import get_activation_dict, get_layer_activations, project_layer_activations_to_input

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
    scam = np.zeros(np.shape(delta_projected[1:]))

    for c in range(channels):
        scam += delta_projected[c,:,:]

    scam /= np.max(np.abs(scam))
    return scam
