import subprocess
import json
import os

def start_training(train_setup,
                   nt_combinations=None,
                   netG="resnet_9blocks",
                   base_data_dir="/nrs/funke/ecksteinn/synister_experiments/cycle_attribution/data_png",
                   continue_train=False):

    checkpoint_base = "/nrs/funke/ecksteinn/synister_experiments/cycle_attribution/checkpoints"

    nt_list = ["gaba", "acetylcholine", "glutamate", 
               "serotonin", "octopamine", "dopamine"]

    base_cmd = "~/singularity/run_lsf -q gpu_any python -u train.py" +\
               " --dataroot {} --name {} --input_nc 1 --output_nc 1 --netG {} --load_size 128 --crop_size 128 --checkpoints_dir /nrs/funke/ecksteinn/synister_experiments/cycle_attribution/checkpoints"

    if continue_train:
        base_cmd += " --continue_train"

    if nt_combinations is None:
        nt_combinations = []
        for ntA in nt_list:
            for ntB in nt_list:
                if ntA != ntB:
                    nt_combinations.append((ntA, ntB))

    for nt_combination in nt_combinations:
        ntA = nt_combination[0]
        ntB = nt_combination[1]

        nts = [ntA, ntB]
        dataroot = os.path.join(base_data_dir, "synister_{}_{}".format(ntA, ntB))
        if not os.path.exists(dataroot):
            dataroot = os.path.join(base_data_dir, "synister_{}_{}".format(ntB, ntA))
            nts = [ntB, ntA]
        assert(os.path.exists(dataroot))
        aux_class_a = nt_list.index(nts[0])
        aux_class_b = nt_list.index(nts[1])
        train_setup_name = "train_{}_{}_s{}".format(nts[0], 
                                                    nts[1], 
                                                    train_setup)

        checkpoint_dir = os.path.join(checkpoint_base, 
                                      train_setup_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        json.dump({"aux_class_A": aux_class_a,
                   "aux_class_B": aux_class_b,
                   "dataroot": dataroot,
                   "train_setup": train_setup,
                   "netG": netG},
                    open(os.path.join(checkpoint_dir, "train_config.json"), "w+"))

        cmd = base_cmd.format(dataroot,
                              train_setup_name,
                              netG
                              )
        subprocess.Popen(cmd, 
                         shell=True) 

if __name__ == "__main__":
    start_training(train_setup=1112, nt_combinations=[("gaba", "acetylcholine")])
