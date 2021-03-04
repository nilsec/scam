import numpy as np
from PIL import Image
import torch

def flatten_image(pil_image):
    """
    pil_image: image as returned from PIL Image
    """
    return np.array(pil_image[:,:,0], dtype=np.float32)

def normalize_image(image):
    """
    image: 2D input image
    """
    return (image.astype(np.float32)/255. - 0.5)/0.5

def open_image(image_path, flatten=True, normalize=True):
    im = np.asarray(Image.open(image_path))
    if flatten:
        im = flatten_image(im)
    if normalize:
        im = normalize_image(im)
    return im

def image_to_tensor(image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_tensor = torch.tensor(image, device=device)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    return image_tensor

def save_image(array, image_path):
    array = (array *0.5 + 0.5)*255
    im = Image.fromarray(array)
    im = im.convert('RGB')
    im.save(image_path)
