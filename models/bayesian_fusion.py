
import torch
from torch import nn
import torch.nn.functional as F

### Modèles encodeurs decodeurs

class FusionBayesianEncoderRNN(torch.nn.Module):
    def __init__(self,input_size, hidden_size, num_grulstm_layers, batch_size):
        super(FusionBayesianEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_grulstm_layers = num_grulstm_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True, dtype = torch.float32)

    def forward(self, input, hidden):
      # REMINDER
      # TOTAL input if hot_encoding : [batch_size, length T, length T, dimensionality d]
      # TOTAL input if time_delay : [batch_size, length T - k, time delay k + 1, dimensionality d]
      # For the input of forward function of one GRU cell, suppress dimension 1 (resp. T and T - k)
        #print("INPUT GRU", input.shape,"\n", "\n", "\n", input) = 50, k+1, num_var
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def init_hidden(self,device):
        #[num_layers*num_directions,batch,hidden_size]
        return torch.zeros(self.num_grulstm_layers, self.batch_size, self.hidden_size, device=device)
        #local_gener = torch.Generator(device=device)
        #local_gener.manual_seed(1) # For reproducibility
        #return torch.randn(self.num_grulstm_layers, self.batch_size, self.hidden_size, device=device, generator=local_gener, dtype=torch.float32)

class FusionBayesianDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_grulstm_layers,fc_units, output_size):
        super(FusionBayesianDecoderRNN, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True, dtype=torch.float32)
        self.fc = nn.Linear(hidden_size, fc_units, dtype=torch.float32)
        self.out = nn.Linear(fc_units, output_size, dtype=torch.float32)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = F.relu( self.fc(output) )
        output = self.out(output)
        return output, hidden

class FusionBayesian_Net_GRU(nn.Module):
    def __init__(self, encoder, decoder, target_length, input_dropout, recurrent_dropout, fusion_style, encoding_strategy_univar, k_for_time_delay, device):
        super(FusionBayesian_Net_GRU, self).__init__()
        if fusion_style == 'early' and encoding_strategy_univar == 'hot_encoding' :
          print("Hot encoding is not recommended for early fusion of variables. It would be very heavy to compute.")
          #raise ValueError("Hot encoding is deactivated for early fusion of variables. It would be too heavy to compute.")
        self.encoder = encoder
        self.decoder = decoder
        self.target_length = target_length
        if fusion_style not in ['early', 'late', 'no fusion']:
          raise ValueError("Fusion style should be either 'early', 'late or 'no fusion' (for univariate data).")
        self.fusion_style = fusion_style
        if encoding_strategy_univar not in ['hot_encoding', 'time_delay_embedding', 'no strategy']:
          raise ValueError("Univariate strategy should be either 'hot_encoding', 'time_delay_embedding' or 'no strategy'.")
        if k_for_time_delay < 0 and encoding_strategy_univar == 'time_delay_embedding' :
          raise ValueError("You are using a time delay embedding. Please specify strictly positive integer for k_for_time_delay.")
        self.encoding_strategy_univar = encoding_strategy_univar
        self.k = k_for_time_delay
        self.device = device
        # Two dropout types :
        self.recurrent_dropout = recurrent_dropout
        self.input_dropout = input_dropout
        # Note : seulement adapté pour une seule lstm_layer...

    def forward(self, x):
        sh = len(x.shape)
        batch_size = x.shape[0]
        input_length  = x.shape[1]
        if sh==2:
          x = x.unsqueeze(2)
        var_number = x.shape[2]

        encoder_hidden = self.encoder.init_hidden(self.device)
        if self.encoding_strategy_univar == 'hot_encoding':
          x_bis = torch.zeros(batch_size, input_length, input_length, var_number).to(self.device, non_blocking=True)
          for i in range(batch_size):
            for j in range(input_length):
              x_bis[i,j,j,:] = x[i,j,:]
        elif self.encoding_strategy_univar == 'time_delay_embedding':
          input_length -= self.k
          x_bis = torch.zeros(batch_size, input_length, self.k + 1, var_number).to(self.device, non_blocking=True)
          for j in range(input_length):
            for q in range(self.k + 1):
              x_bis[:,j,q,:] = x[:,j+q,:]


        if self.training: # Activation du dropout si le modèle s'entraîne

          if self.input_dropout>0:
            if self.encoding_strategy_univar != 'no strategy':
              probs_in = torch.full_like(x_bis[:,0:1,:,:], 1.0 - self.input_dropout)
            else:
              probs_in = torch.full_like(x[:,0:1,:], 1.0 - self.input_dropout)
            # We are using Invertible Dropout, so we'll have to rescale.
            # It inflates the values of the surviving neurons to compensate for those we destroyed
            scale_in = 1.0 / (1.0 - self.input_dropout)
            # Création des masques qui remplaceront le dropout :
            mask_in = torch.bernoulli(probs_in).to(self.device, non_blocking=True)
            mask_in = mask_in * scale_in


          if self.recurrent_dropout>0:
            # Pour l'instant, même masque dans l'enc et le déc:
            probs_hidden = torch.full_like(encoder_hidden, 1.0 - self.recurrent_dropout)
            #scale_hidden = 1.0 / (1.0 - self.recurrent_dropout)
            mask_hidden = torch.bernoulli(probs_hidden).to(self.device, non_blocking=True)
            #mask_hidden = (mask_hidden * scale_hidden)*(1-self.recurrent_dropout)
            # Cette ligne ci-dessus nous a permis de résoudre le problème de l'apparition des valeurs manquantes.
            # En voyant que cela fonctionnait, nous avons décidé d'utiliser un décrochage traditionel pour
            # self.recurrent_dropout

        # Les masques sont générés et seront égaux pour toute la passe forward. On peut commencer!
        for ei in range(input_length):
            if self.encoding_strategy_univar != 'no strategy':
              encoder_input = x_bis[:,ei:ei+1,:,:]
            #elif self.encoding_strategy_univar == 'time_delay_embedding':
              #encoder_input = x[:,ei:ei+self.k,:] #Would this be more efficient ?
            else:
              encoder_input = x[:,ei:ei+1,:]
            if self.training:
              if self.input_dropout>0:
                encoder_input = encoder_input * mask_in
              if self.recurrent_dropout>0:
                encoder_hidden = encoder_hidden * mask_hidden

            # Si il y a hot_encoding, l'encodeur va manger des tenseurs de forme : (batch_size, input_length, num_variables)
            # cela signifie que chaque tenseur d'entrée est bien de forme "pseudo-one-hot" !
            # c'était juste pour y passer le masque.
            if self.encoding_strategy_univar != 'no strategy':
              encoder_input = encoder_input.squeeze(1)

            encoder_output, encoder_hidden = self.encoder(encoder_input, encoder_hidden)


        decoder_input = x[:,-1,:].unsqueeze(1).to(torch.float32) # first decoder input= last element of input sequence
      ##  if self.training and self.recurrent_dropout>0:
        ##  encoder_hidden = encoder_hidden * mask_hidden
       # if self.training and self.input_dropout>0:
        #  decoder_input = decoder_input * mask_in
        decoder_hidden = encoder_hidden

        outputs = torch.zeros([batch_size, self.target_length, x.shape[2]]).to(self.device, non_blocking=True)

        # Mauvaise nouvelle... les masques doivent être différents pour la partie décodeur!

        if self.training :
          if self.input_dropout>0:
            probs_in = torch.full_like(x[:,0:1,:], 1.0 - self.input_dropout, dtype=torch.float32)
            scale_in = 1.0 / (1.0 - self.input_dropout)
            mask_in = torch.bernoulli(probs_in).to(self.device, non_blocking=True)
            mask_in = mask_in * scale_in
          if self.recurrent_dropout>0:
            probs_hidden = torch.full_like(encoder_hidden, 1.0 - self.recurrent_dropout)
            mask_hidden = torch.bernoulli(probs_hidden).to(self.device, non_blocking=True)


        for di in range(self.target_length):
            if self.training:
              if self.input_dropout>0:
                decoder_input = decoder_input * mask_in
              if self.recurrent_dropout>0:
                decoder_hidden = decoder_hidden * mask_hidden

            decoder_gru_output, decoder_hidden = self.decoder.gru(decoder_input, decoder_hidden)

            decoder_output_transformed = F.relu( self.decoder.fc(decoder_gru_output) )
            decoder_output_transformed = self.decoder.out(decoder_output_transformed)

            decoder_input = decoder_output_transformed
            outputs[:,di:di+1,:] = decoder_output_transformed

        return outputs
