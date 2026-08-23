import torch
from torch import nn
import torch.nn.functional as F
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(0)

def monte_carlo_dropout(model, loader, batch_size, N_output, s):
  # s sup ou egal à 1
  batch_number = len(loader)
  num_var = loader.dataset.num_variables

  outputs = torch.zeros(batch_number, batch_size, N_output, num_var, 2).to(device) # dernières dimensions: moyenne, écart-type
  if s>1 :
    model.Training = True #enables dropout
  else :
    model.Training = False #disables dropout

  met = 0

  for i, data in enumerate(loader, 0):
    inputs, target = data
    inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
    target = torch.tensor(target, dtype=torch.float32).to(device)
    for j in range(s):
      with torch.no_grad(): # no gradient computation, otherwise it'll enable "optimizer_step" + train model (if s=2 or more)
        outputs_temp = model(inputs).to(device) # it's a torch.tensor, but one dimension too much
        for v in range(num_var) :
          outputs[i,:,:,v:v+1,0] += outputs_temp[:,:,v:v+1]
          outputs[i,:,:,v:v+1,1] += outputs_temp[:,:,v:v+1]**2
    mean_batch = outputs[i,:,:,:,0]/s
    outputs[i,:,:,:,0] = mean_batch
    if s==1:
      outputs[i,:,:,:,1] = torch.zeros(batch_size, N_output, num_var) 
    else:
      outputs[i,:,:,:,1] = outputs[i,:,:,:,1]/(s-1) - mean_batch**2
    outputs[i,:,:,:,1] = torch.sqrt(outputs[i,:,:,:,1])

    batch_m = outputs[i,:,:,:,0]
    batch_std = outputs[i,:,:,:,1]

    lower_bound = batch_m - 1.96*batch_std 
    upper_bound = batch_m + 1.96*batch_std

    covered = (target >= lower_bound) & (target <= upper_bound)
    good_in_batch = (covered).sum().item()

    met += good_in_batch

  met_score = 100 * met / (batch_number * batch_size * N_output * num_var)
  print(f"La cible est incluse dans l'intervalle de confiance à 95% donné par les simulations dans : {met_score:.2f} % des cas")
  return outputs
