import os
import subprocess
from shutil import copyfile

def start_testing(nt_combinations,
                  test_nt,
                  dataroot,
                  results_dir,
                  netG="resnet_9blocks",
                  aux_checkpoint="/nrs/funke/ecksteinn/synister_experiments/gan/02_train/setup_t0/model_checkpoint_499000",
                  num_test=500,
                  train_setup=2):

    base_cmd = "~/singularity/run_lsf python -u test.py --model test --no_dropout --results_dir {} --dataroot {} --checkpoints_dir {} --name {} --model_suffix {} --num_test {} --aux_checkpoint {} --aux_input_size 128 --aux_net vgg2d --aux_input_nc 1 --num_threads 1 --verbose"
    base_dir = "/nrs/funke/ecksteinn/synister_experiments/cycle_attribution"

    if test_nt == nt_combinations[0]:
        a_or_b = "A"
    elif test_nt == nt_combinations[1]:
        a_or_b = "B"

    nt_list = ["gaba", "acetylcholine", "glutamate",
               "serotonin", "octopamine", "dopamine"]

    train_setup_name = "train_{}_{}_s{}".format(nt_combinations[0],
                                                nt_combinations[1],
                                                train_setup)
    checkpoint_dir = base_dir + "/checkpoints"
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    cmd = base_cmd.format(results_dir,
                          dataroot,
                          checkpoint_dir,
                          train_setup_name,
                          "_" + a_or_b,
                          num_test,
                          aux_checkpoint)

    subprocess.Popen(cmd,
                     shell=True)
    #subprocess.check_call(cmd,shell=True)
    #print(cmd)
def get_combinations(train_setup=2):
    base_dir = "/nrs/funke/ecksteinn/synister_experiments/cycle_attribution/checkpoints"
    nt_list = ["gaba", "acetylcholine", "glutamate",
               "serotonin", "octopamine", "dopamine"]

    nt_combinations = []
    for ntA in nt_list:
        for ntB in nt_list:
            if ntA != ntB:
                checkpoint_dir = base_dir + "/train_{}_{}_s{}".format(ntA, ntB, train_setup)
                if os.path.exists(checkpoint_dir + "/web"):
                    nt_combinations.append((ntA, ntB))

    return nt_combinations


def create_results_data_dir(result_image_dir, base_data_dir, cycle):
    fake_images = [os.path.join(result_image_dir,f) for f in os.listdir(result_image_dir) if f.endswith("fake.png")]
    cycle_dir = result_image_dir + "/cycle_{}".format(cycle)
    os.makedirs(cycle_dir)
    for im in fake_images:
        copyfile(im, cycle_dir + "/" + os.path.basename(im))

    return cycle_dir 


def start_testing_cycles(nt_combinations,
                         test_nt,
                         n_cycles,
                         dataroot,
                         results_dir,
                         train_setup):

    for cycle in range(n_cycles):
        start_testing(nt_combinations=nt_combinations,
                      test_nt=test_nt,
                      dataroot=dataroot,
                      results_dir=results_dir)

        image_dir = results_dir + "/train_{}_{}_s{}/test_latest/images".format(nt_combinations[0], nt_combinations[1], train_setup)

        dataroot = create_results_data_dir(image_dir, 
                                           image_dir, cycle)
        results_dir = dataroot + "/results"

def test_all_cycles(test_setup, n_cycles=5, train_setup=2):
    combinations = get_combinations()
    for comb in combinations:
        for nt in comb:
            base_dir = "/nrs/funke/ecksteinn/synister_experiments/cycle_attribution"
            direction = ["A","B"][comb.index(nt)]
            dataroot = base_dir + "/data_png/synister_{}_{}/train{}".format(comb[0],
                                                                            comb[1],
                                                                            direction)

            results_dir = "/nrs/funke/ecksteinn/synister_experiments/cycle_attribution/results/test_{}_{}_{}_c{}".format(comb[0], 
                                                                                                                         comb[1], 
                                                                                                                         direction,
                                                                                                                         test_setup)

            start_testing_cycles(nt_combinations=comb,
                                 test_nt=nt,
                                 n_cycles=n_cycles,
                                 dataroot=dataroot,
                                 results_dir=results_dir,
                                 train_setup=train_setup)
def test_all(train_setup, test_setup):
    combinations = get_combinations()
    for comb in combinations:
        for nt in comb:
            base_dir = "/nrs/funke/ecksteinn/synister_experiments/cycle_attribution"
            direction = ["A","B"][comb.index(nt)]
            dataroot = base_dir + "/data_png/synister_{}_{}/train{}".format(comb[0],
                                                                            comb[1],
                                                                            direction)

            results_dir = "/nrs/funke/ecksteinn/synister_experiments/" +\
                    "cycle_attribution/results/test_{}_{}_{}_t{}".format(comb[0], 
                                                                         comb[1], 
                                                                         direction,
                                                                         test_setup)
            start_testing(comb,
                          nt,
                          dataroot,
                          results_dir,
                          train_setup=train_setup)

if __name__ == "__main__":
    test_all(2, 0)
