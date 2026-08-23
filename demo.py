#pip install tslearn
#pip install -r requirements.txt
#pip install torch-timeseries

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
#from torch.utils.data import DataLoader
import random
from tslearn.metrics import dtw, dtw_path
import matplotlib.pyplot as plt
import warnings
import warnings; warnings.simplefilter('ignore')

from torch_timeseries.dataset import Traffic

from src.loader import OurDataset, create_time_series_dataset, prepare_univariate_data
from src.visual import visualiser_avec_mcd_loaderformat
from src.monte_carlo_simulations import monte_carlo_dropout
from models.bayesian_fusion import FusionBayesianEncoderRNN, FusionBayesianDecoderRNN, FusionBayesian_Net_GRU
from loss import soft_dtw, path_soft_dtw, dilate_loss
from loss.dilate_loss import dilate_loss
from models.train import train_model2

epochs_num_dil = 0
epochs_num_mse = 0
N_input=168
N_output=24
batch_size = 182
num_var = 1
gamma = 0.1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(0)

#load data
trafic = Traffic("./data")
donnees_trafic = trafic.data[:-24*3,0:200].astype(np.float32) #découpage par tranches de 8 jours pendant deux ans, il fallait supprimer trois jours
donnees_trafic_rangees = donnees_trafic.transpose().flatten()

scaler_traffic, trafic_train, trafic_test, trafic_val = prepare_univariate_data(
                        series_data = donnees_trafic_rangees, nav = N_input, nap = N_output,
                        train_size=5460*2, test_size = 1820*2, val_size = 1820*2,
                        create_val_loader=True, batch_size=182, num_workers=4)

encoder2 = FusionBayesianEncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=182).to(device)
decoder2 = FusionBayesianDecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1,fc_units=16, output_size=1).to(device)
var_net_gru_dilate = FusionBayesian_Net_GRU(encoder2, decoder2, 24, 0.1, 0.1, fusion_style = "early", encoding_strategy_univar = 'time_delay_embedding', k_for_time_delay = 1, device=device).to(device)

encoder = FusionBayesianEncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=182).to(device)
decoder = FusionBayesianDecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1,fc_units=16, output_size=1).to(device)
var_net_gru_mse = FusionBayesian_Net_GRU(encoder, decoder, 24, 0.1, 0.1, fusion_style = "early", encoding_strategy_univar = 'time_delay_embedding', k_for_time_delay = 1, device=device).to(device)

#load pre-trained models                      
model_path_dil = './models/model_dilate.pt'
model_path_mse = './models/model_mse.pt'

state_dict_dil = torch.load(model_path_dil, map_location=device)
state_dict_mse = torch.load(model_path_mse, map_location=device)

var_net_gru_dilate.load_state_dict(state_dict_dil)
var_net_gru_mse.load_state_dict(state_dict_mse)

var_net_gru_dilate.eval()
var_net_gru_mse.eval()

# Just to check :
def test_dropout(net,input, activate_dropout=True):
  input = torch.tensor(input, dtype=torch.float32).to(device)
  outputs = []

  if activate_dropout :
      net.train()
      with torch.no_grad():
        for i in range(3):
          out = net(input).to(device)
          outputs.append(out[0].detach().cpu().numpy())
      ret = (outputs[0] != outputs[1]).any() or (outputs[0] != outputs[2]).any() or (outputs[1] != outputs[2]).any()
      if ret :
        print(f'Dropout is working : your model is not deterministic in trainig mode')
      else :
        print(f'Dropout seems not to be working : model seems to be deterministic in trainig mode')
      return ret

  else:
    net.eval()
    with torch.no_grad():
      for i in range(3):
        out = net(input)
        outputs.append(out.detach().cpu().numpy())
    ret = (outputs[0] == outputs[1]).all() and (outputs[0] == outputs[2]).all()
    print(f'Model is deterministic in evaluation mode' if ret else f'Model is not deterministic in evaluation mode')
    return ret

with torch.no_grad():
  for i, data in enumerate(trafic_test, 0):
      inputs, target = data 
      inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
      target = torch.tensor(target, dtype=torch.float32).to(device)
      plt.plot(target[180].detach().cpu().numpy(), color = 'pink')
      test_dropout(net=var_net_gru_dilate, input=inputs, activate_dropout = True)
      for j in range(4):
        res = var_net_gru_dilate(torch.tensor(inputs, dtype=torch.float32).to(device))
        plt.plot(res[180].detach().cpu().numpy())
      plt.savefig(f'.variations_test.png', dpi=300, bbox_inches="tight")
      plt.close()
      print(f"Graphique sauvegardé sous : .variations_test.png")
      if i == 0 :
        break

# If you want to retrain one of the models, please change the number of epochs in the first line of variable epochs_num to a strictly positive integer. Be careful, the training is quite long ! :)
# Note that it will also terminate demo.py execution, so you won't be able to train the other model in the same run, nor see the results of the MC simulations. Working on it ! 
train_model2(var_net_gru_dilate, trafic_train, trafic_test, loss_type='dilate',learning_rate=0.001, epochs=epochs_num_dil, gamma=gamma, print_every=10, eval_every=50, verbose=1, alpha = 0.8)
train_model2(var_net_gru_mse, trafic_train, trafic_test, loss_type='mse',learning_rate=0.001, epochs=epochs_num_mse, gamma=gamma, print_every=10, eval_every=50, verbose=1, alpha = 0.8)

# Time to see our results :
print("visualisation pour le réseau entraîné sur DILATE:")
sim_dil = monte_carlo_dropout(var_net_gru_dilate, trafic_val, batch_size=182, N_output=24, s=100)
variables = ["taux trafic"]
visualiser_avec_mcd_loaderformat(trafic_val,
                                sim_dil, N_input=168, N_output=24,
                                which_batch=7, which_data=3, variables=variables, title='Monte Carlo Dropout Results for DILATE-trained model on traffic data. 100 simulations.',
                                save_path='.monte_carlo_dropout_dilate.png', reduce_input_window=True)

print("Visualisation pour le réseau entraîné sur MSE:")
sim_mse = monte_carlo_dropout(var_net_gru_mse, trafic_val, batch_size=182, N_output=24, s=100)
visualiser_avec_mcd_loaderformat(trafic_val,
                                sim_mse, N_input=168, N_output=24,
                                which_batch=7, which_data=3, variables=variables, title='Monte Carlo Dropout Results for MSE-trained model on traffic data. 100 simulations.',
                                save_path='.monte_carlo_dropout_mse.png', reduce_input_window=True)
