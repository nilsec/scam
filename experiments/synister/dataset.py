from scam.dataset import create_dataset
from scam.utils import get_all_pairs

import os
    
nts = ["gaba", "acetylcholine", "glutamate", "serotonin", "octopamine", "dopamine"]

def parse_pairs():
    base_dir = "/nrs/funke/ecksteinn/synister_experiments/cycle_attribution/results"
    pairs = get_all_pairs(nts)

    pair_to_images = {}
    for pair in pairs:
        for direction in ["A", "B"]:
            image_dir = base_dir + f"/test_{pair[0]}_{pair[1]}_{direction}_t0/train_{pair[0]}_{pair[1]}_s2/test_latest/images"
            images = os.listdir(image_dir)
            real = [os.path.join(image_dir,im) for im in images if "real.png" in im]
            fake = [os.path.join(image_dir,im) for im in images if "fake.png" in im]

            if direction == "A":
                pair_name = pair
            else:
                pair_name = pair[::-1]

            paired_images = []
            for r in real:
                for f in fake:
                    if r.split("/")[-1].split("_")[0] == f.split("/")[-1].split("_")[0]:
                        paired_images.append((r,f))
                        break

            pair_to_images[pair_name] = paired_images

    return pair_to_images

def create_synister_dataset():
    net_module = "Vgg2D"
    checkpoint = "/nrs/funke/ecksteinn/synister_experiments/gan/02_train/setup_t0/model_checkpoint_499000"
    input_shape = (128,128)
    input_nc = 1

    thr = 0.8
    pair_to_images = parse_pairs()

    for pair, images in pair_to_images.items():
        real_class = nts.index(pair[0])
        fake_class = nts.index(pair[1])
        create_dataset(f"experiments/synister/data/{pair[0]}_{pair[1]}", images, real_class, fake_class, net_module, 
                       checkpoint, input_shape, input_nc, real_thr=thr, fake_thr=thr)

if __name__ == "__main__":
    create_synister_dataset()

