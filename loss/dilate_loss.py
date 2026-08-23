import torch
from . import soft_dtw
from . import path_soft_dtw

def dilate_loss(outputs, targets, alpha, gamma, device):
    # outputs, targets: shape (batch_size, N_output, 1)
    batch_size, N_output = outputs.shape[0:2]

    loss_shape = 0
    softdtw_batch = SoftDTWBatch.apply
    D = torch.zeros((batch_size, N_output,N_output)).to(device, non_blocking = True)
    for k in range(batch_size):
      #for i in range(num_variables):
      Dk = pairwise_distances(targets[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1))
      D[k:k+1,:,:] = Dk
    loss_shape = softdtw_batch(D,gamma)

    path_dtw = PathDTWBatch.apply
    path = path_dtw(D,gamma)
    Omega =  pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device, non_blocking = True)
    loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output)
    loss = alpha*loss_shape+ (1-alpha)*loss_temporal
    return loss, loss_shape, loss_temporal
