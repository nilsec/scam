import subprocess
import itertools
import json
import os

def start_training(train_setup,
                   data_roots,
                   netG="resnet_9blocks",
                   in_size=128,
                   continue_train=False):

    for data_root in data_roots:
        base_cmd = "~/singularity/run_lsf -q gpu_any python -u train.py" +\
                   " --dataroot {} --name {} --input_nc 1 --output_nc 1 --netG {} --load_size {} --crop_size {} --checkpoints_dir {} --display_id 0"

        if continue_train:
            base_cmd += " --continue_train"

        train_setup_name = "train_s{}".format(train_setup)
        checkpoint_dir = os.path.join(os.path.join(data_root, "setups"), 
                                      train_setup_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        cmd = base_cmd.format(data_root,
                              train_setup_name,
                              netG,
                              in_size,
                              in_size,
                              checkpoint_dir)
        subprocess.Popen(cmd, 
                         shell=True) 


def train_fade():
    datasets = {"1a": ["0_1", "0_2", "1_2"],
                "1b": ["0_1", "0_2", "1_2"],
                "1c": ["0_1", "0_2", "1_2"],
                "1d": ["0_1"],
                "2a": ["0_1"],
                "2b_s": ["0_1"],
                "2c_s": ["0_1"],
                "2d_s": ["0_1"]}

    data_roots = []
    for d, k in datasets.items():
        for j in k:
            data_roots.append(f"/nrs/funke/ecksteinn/soma_data/{d}/cycle_gan/{j}")

    start_training(train_setup=0, 
                   data_roots=data_roots)

def train_mnist():
    datasets = [f"{i}_{j}" for i,j in list(itertools.combinations([i for i in range(10)], 2))]
    data_roots = []

    for d in datasets:
        data_roots.append(f"/nrs/funke/ecksteinn/mnist_png/training/cycle_gan/{d}")

    print(data_roots)

    start_training(train_setup=0, 
                   data_roots=data_roots,
                   in_size=28)


if __name__ == "__main__":
    train_mnist()
