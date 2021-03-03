import importlib
import torch
import torch.nn.functional as F
from torch_receptive_field import receptive_field, receptive_field_for_unit

from scam.utils import image_to_tensor

def init_network(checkpoint_path=None, input_size=128, net_module="Vgg2D", input_nc=1, gpu_ids=[], eval_net=True, require_grad=False):
    """
    checkpoint_path: Path to train checkpoint to restore weights from

    input_nc: input_channels for aux net

    aux_net: name of aux net
    """
    net_mod = importlib.import_module(f"networks.{net_module}")
    net_class = getattr(net_mod, f'{net_module}')
    net = net_class(input_size=(input_size, input_size), input_channels=input_nc)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    if not require_grad:
        # Freeze parameters
        for param in net.parameters():
            param.requires_grad = False
    if eval_net:
        # Disable batch norm and dropout in aux
        net.eval()

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
    return net

def run_inference(net, im):
    """
    Net: network object
    input_image: Normalized 2D input image.
    """
    im_tensor = image_to_tensor(im)
    class_probs = F.softmax(net(im_tensor), dim=1)
    return class_probs
