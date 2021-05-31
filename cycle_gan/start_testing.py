import os
import subprocess
import itertools
from shutil import copyfile

def start_testing(class_pair,
                  test_class,
                  checkpoints_dir,
                  data_root,
                  results_dir,
                  aux_checkpoint,
                  aux_output_classes,
                  aux_downsample_factors=[(2,2),(2,2),(2,2),(2,2)],
                  aux_net="vgg2d",
                  input_size=128,
                  netG="resnet_9blocks",
                  num_test=500):

    # Workaround for cycle_gan convention
    name = os.path.basename(checkpoints_dir)
    checkpoints_dir = os.path.dirname(checkpoints_dir)

    base_cmd = "~/singularity/run_lsf -q gpu_any python -u cycle_gan/test.py --model test --no_dropout --results_dir {} --dataroot {} --checkpoints_dir {} --name {} --model_suffix {} --num_test {} --aux_checkpoint {} --aux_input_size {} --aux_net {} --aux_input_nc 1 --num_threads 1 --verbose --aux_output_classes {} --aux_downsample_factors '{}'"

    if test_class == class_pair[0]:
        a_or_b = "A"
    elif test_class == class_pair[1]:
        a_or_b = "B"

    data_root = os.path.join(data_root, "train" + a_or_b)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)


    aux_downsample_factors_string = ""
    for fac in aux_downsample_factors:
        for dim in fac:
            aux_downsample_factors_string+=f"{dim},"
        aux_downsample_factors_string=aux_downsample_factors_string[:-1] + "x"
    aux_downsample_factors_string = aux_downsample_factors_string[:-1]

    cmd = base_cmd.format(results_dir,
                          data_root,
                          checkpoints_dir,
                          name,
                          "_" + a_or_b,
                          num_test,
                          aux_checkpoint,
                          input_size,
                          aux_net,
                          aux_output_classes,
                          aux_downsample_factors_string)

    subprocess.Popen(cmd,
                     shell=True)

def test_all_fade():
    level_to_classes = {"1a": [0,1,2],
			"1b": [0,1,2],
			"1c": [0,1,2],
			"1d": [0,1],
			"2a": [0,1],
			"2b_s": [0,1],
			"2c_s": [0,1],
			"2d_s": [0,1]}

    vgg_winner = {"1a": 14,
                  "1b": 72,
                    "1c": 63,
                    "1d": 83,
                    "2a": 7,
                    "2b_s": 58,
                    "2c_s": 87,
                    "2d_s": 81}

    res_winner = {"1a": 37,
                  "1b": 87,
                    "1c": 30,
                    "1d": None, # 58%
                    "2a": 24,
                    "2b_s": None, # 62%
                    "2c_s": None, # 61% 
                    "2d_s": None} # 58%

    nets = {"vgg": vgg_winner, "res": res_winner}

    for net in nets:
        for level, classes in level_to_classes.items():
            class_pairs = [[i,j] for i,j in list(itertools.combinations(classes, 2))]
            for class_pair in class_pairs:
                dataset = f"{class_pair[0]}_{class_pair[1]}"
                data_root = f"/nrs/funke/ecksteinn/soma_data/{level}/cycle_gan/{dataset}"
                checkpoints_dir = os.path.join(data_root, "setups/train_s0/train_s0")

                epoch = nets[net][level]
                if epoch is None:
                    continue
                aux_checkpoint = f"/groups/funke/home/ecksteinn/Projects/scam/experiments/fade/classifiers/{level}_{net}/epoch_{epoch}"
                aux_output_classes = len(classes)
                aux_downsample_factors = [(2,2),(2,2),(2,2),(2,2)]

                aux_net = net
                if net == "vgg":
                    aux_net = "vgg2d"

                for test_class in class_pair:
                    print(net, level, class_pair, test_class)
                    results_dir = os.path.join(data_root, f"results/{net}_{test_class}")

                    start_testing(class_pair,
                                  test_class,
                                  checkpoints_dir,
                                  data_root,
                                  results_dir,
                                  aux_checkpoint,
                                  aux_output_classes,
                                  aux_downsample_factors=aux_downsample_factors,
                                  aux_net=aux_net,
                                  input_size=128,
                                  netG="resnet_9blocks",
                                  num_test=1000)
                                     
def test_all_mnist():
    vgg_winner = 92
    res_winner = 71 

    nets = {"vgg": vgg_winner, "res": res_winner}
    classes = [k for k in range(10)]

    for net in nets:
        class_pairs = [[i,j] for i,j in list(itertools.combinations(classes, 2))]
        for class_pair in class_pairs:
            dataset = f"{class_pair[0]}_{class_pair[1]}"
            data_root = f"/nrs/funke/ecksteinn/mnist_png/training/cycle_gan/{dataset}"
            checkpoints_dir = os.path.join(data_root, "setups/train_s0/train_s0")
            epoch = nets[net]
            aux_checkpoint = f"/groups/funke/home/ecksteinn/Projects/scam/experiments/mnist/classifiers/{net}/epoch_{epoch}"
            aux_output_classes = len(classes)

            aux_downsample_factors = [(2,2),(2,2),(1,1),(1,1)]

            aux_net = net
            if net == "vgg":
                aux_net = "vgg2d"

            for test_class in class_pair:
                print(net, class_pair, test_class)
                results_dir = os.path.join(data_root, f"results/{net}_{test_class}")

                start_testing(class_pair,
                              test_class,
                              checkpoints_dir,
                              data_root,
                              results_dir,
                              aux_checkpoint,
                              aux_output_classes,
                              aux_downsample_factors=aux_downsample_factors,
                              aux_net=aux_net,
                              input_size=28,
                              netG="resnet_9blocks",
                              num_test=1000)
                                 
if __name__ == "__main__":
    test_all_mnist()

    """
    class_pair = [0,1]
    test_class = 0
    checkpoints_dir = "/nrs/funke/ecksteinn/soma_data/1a/cycle_gan/0_1/setups/train_s0/train_s0"
    data_root = "/nrs/funke/ecksteinn/soma_data/1a/cycle_gan/0_1"
    results_dir = "/nrs/funke/ecksteinn/soma_data/1a/cycle_gan/0_1/results"
    aux_checkpoint = "/groups/funke/home/ecksteinn/Projects/scam/experiments/fade/classifiers/1a_vgg/epoch_99"
    aux_output_classes = 3
    aux_downsample_factors = [(2,2),(2,2),(2,2),(2,2)]

    start_testing(class_pair,
                  test_class,
                  checkpoints_dir,
                  data_root,
                  results_dir,
                  aux_checkpoint,
                  aux_output_classes,
                  aux_downsample_factors=aux_downsample_factors,
                  aux_net="vgg2d",
                  input_size=128,
                  netG="resnet_9blocks",
                  num_test=10)
    """

