import os
import numpy as np
import json

from scam.utils import open_image, image_to_tensor
from networks import init_network
from scam import get_scammed

def get_nth_sorted_image(idx, nt_A, nt_B, a_to_b, 
                         test_setup=0, flatten=True, 
                         normalize=True):
    
    test_dir = get_test_dir(nt_A,
                            nt_B,
                            a_to_b,
                            test_setup)

    real_images, fake_images, aux_info = parse_prediction(test_dir)

    sorted_all = sort_images(real_images,
                             fake_images,
                             aux_info,
                             nt_A,
                             nt_B,
                             a_to_b)
    
    im_index = sorted_all[idx][-1]
    path_real = real_images[im_index]
    path_fake = fake_images[im_index]
    im_real = open_image(path_real, flatten, normalize)
    im_fake = open_image(path_fake, flatten, normalize)
    return im_real, im_fake

def read_aux(aux_path):
    with open(aux_path, "r") as f:
        aux_dict = json.load(f)
    return aux_dict

def parse_prediction(predict_dir):
    files_in_dir = os.listdir(predict_dir)
    real_images_in_dir = {int(f.split("_")[0]): os.path.join(predict_dir, f) for f in files_in_dir if f.endswith("real.png")}
    fake_images_in_dir = {int(f.split("_")[0]): os.path.join(predict_dir, f) for f in files_in_dir if f.endswith("fake.png")}

    aux_info_in_dir = {int(f.split("_")[0]): os.path.join(predict_dir, f) for f in files_in_dir if f.endswith(".json")}
    aux_info_in_dir = {v: read_aux(aux_info_in_dir[v]) for v in aux_info_in_dir.keys()}
    
    return real_images_in_dir, fake_images_in_dir, aux_info_in_dir

def get_test_dir(nt_A, nt_B, a_to_b, test_setup, prefix="t", train_setup=2):
    base_dir = "/nrs/funke/ecksteinn/synister_experiments/cycle_attribution"
    result_dir = "results/test_{}_{}_{}_{}{}/train_{}_{}_s{}/test_latest/images".format(nt_A, nt_B, a_to_b, 
                                                                                        prefix,test_setup, nt_A, nt_B, 
                                                                                        train_setup)

    test_dir = base_dir + "/" + result_dir
    return test_dir

def sort_images(real_images,
                fake_images,
                aux_info,
                nt_A,
                nt_B,
                a_to_b,
                real_min=0.8):
    
    nt_list = ["gaba", "acetylcholine", "glutamate", "serotonin", "octopamine", "dopamine"]
    aux_class_a = nt_list.index(nt_A)
    aux_class_b = nt_list.index(nt_B)
    
    if a_to_b == "A":
        real = aux_class_a
        fake = aux_class_b
    elif a_to_b == "B":
        real = aux_class_b
        fake = aux_class_a
    else:
        raise ValueError("Class not understood")
        
    data_all = []
    
    for v, aux in aux_info.items():
        real_aux = aux["aux_real"][real]
        fake_aux = aux["aux_fake"][fake]
        
        if real_aux < real_min:
            continue
            
        data_all.append((real_aux, fake_aux, v))
        
    sorted_all = sorted(data_all, key=lambda triplet: triplet[1], reverse=True)
    return sorted_all

if __name__ == "__main__":
    vgg_checkpoint_path = "/nrs/funke/ecksteinn/synister_experiments/gan/02_train/setup_t0/model_checkpoint_499000"
    net = init_network(eval_net=True, require_grad=False)
    input_shape = (1,128,128)
    real, fake = get_nth_sorted_image(0, "gaba", "glutamate", "A", 0)
    get_scammed(real, fake, 0, 1, net, input_shape, 24)
