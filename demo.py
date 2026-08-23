pip install tslearn
pip install -r requirements.txt
pip install torch-timeseries

from src import *
from models import *
from loss import *

epochs_num = 0
N_input=168
N_output=24
batch_size = 182
num_var = 1

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

state_dict_dil = torch.load(model_path_dil)
state_dict_mse = torch.load(model_path_mse)

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
      if i == 0 :
        break

# If you want to retrain the model, please change the number of epochs in the first line
train_model2(var_net_gru_dil, trafic_train, trafic_test, loss_type='dilate',learning_rate=0.001, epochs=epochs_num, gamma=gamma, print_every=10, eval_every=50, verbose=1, alpha = 0.8)
train_model2(var_net_gru_mse, trafic_train, trafic_test, loss_type='mse',learning_rate=0.001, epochs=epochs_num, gamma=gamma, print_every=10, eval_every=50, verbose=1, alpha = 0.8)

# Time to see our results :
print("visualisation pour le réseau entraîné sur DILATE:")
sim_dil = monte_carlo_dropout(var_net_gru_dil, trafic_val, batch_size=182, N_output=24, s=100)
variables = ["taux trafic"]
visualiser_avec_mcd_loaderformat(trafic_val,
                                sim_dil, N_input=168, N_output=24,
                                which_batch=7, which_data=3, variables=variables,
                                reduce_input_window=True)

print("Visualisation pour le réseau entraîné sur MSE:")
sim_mse = monte_carlo_dropout(var_net_gru_mse, trafic_val, batch_size=182, N_output=24, s=100)
visualiser_avec_mcd_loaderformat(trafic_val,
                                sim_mse, N_input=168, N_output=24,
                                which_batch=7, which_data=3, variables=variables,
                                reduce_input_window=True)
