import torch
from functools import partial

def hook_fn(in_grads, out_grads, m, i, o):
  for grad in i:
    try:
      in_grads.append(grad)
    except AttributeError: 
      pass

  for grad in o:  
    try:
      out_grads.append(grad.cpu().numpy())
    except AttributeError: 
      pass
    
def get_gradients_from_layer(net, x, y, layer_number, normalize=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xx = torch.tensor(x, device=device).unsqueeze(0)
    yy = torch.tensor([y], device=device)
    xx = xx.unsqueeze(0)
    in_grads = []
    out_grads = []
    for param in net.features.parameters():
        param.requires_grad = True
    net.features[layer_number].register_backward_hook(partial(hook_fn, in_grads, out_grads))
    out = net(xx)
    out[0][y].backward()
    grad = out_grads[0]
    if normalize:
        max_grad = np.max(np.abs(grad))
        if max_grad>10**(-12):
            grad /= max_grad 
        else:
            grad = np.zeros(np.shape(grad))

    return grad
