# parameters
batch_size = 50
N_input = 90
N_output = 55
sigma = 0.01
gamma = 0.01

losses_train, losses_train_shape, losses_train_temporal = [], [], []
losses_mse, losses_dtw, losses_tdi = [], [], []

# We added a learning rate scheduler
def train_model2(net,trainloader, testloader, loss_type, learning_rate, epochs=1000, gamma = 0.001,
                print_every=50,eval_every=50, verbose=1, Lambda=1, alpha=0.75, sched_patience = 10):

    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
    criterion = torch.nn.MSELoss()
    net.train()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5)

    shape_stock, temporal_stock = torch.tensor(0.), torch.tensor(0.)

    for epoch in range(epochs):
        gloss,gloss_shape,gloss_temporal = torch.tensor(0., device=device),torch.tensor(0., device=device),torch.tensor(0., device=device)
        for i, data in enumerate(trainloader, 0):
            inputs, target = data 
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device, non_blocking = True)
            target = torch.tensor(target, dtype=torch.float32).to(device, non_blocking = True)

            if epoch == 0 and i==0:
              batch_size, N_output = target.shape[0:2]
              num_variables = target.shape[2] if len(target.shape) > 2 else 1

            outputs = net(inputs)
            loss_mse,loss_shape,loss_temporal = torch.tensor(0),torch.tensor(0),torch.tensor(0)


            if (loss_type=='dilate'):
                a, b, c = torch.tensor(0., device=device), torch.tensor(0., device=device), torch.tensor(0., device=device)
                for v in range(num_variables) :
                  target_temp, outputs_temp = target[:,:,v:v+1], outputs[:,:,v:v+1]
                  a1,b1,c1 = dilate_loss(target_temp,outputs_temp,alpha, gamma, device)
                  a += a1
                  b += b1
                  c += c1
                loss, loss_shape, loss_temporal = a/num_variables, b/num_variables, c/num_variables
                gloss += loss.item()
                gloss_shape += loss_shape.item()
                gloss_temporal += loss_temporal.item()

            elif (loss_type=='mse'):
                loss = criterion(target,outputs)
                gloss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        gloss = gloss / len(trainloader)

        if (loss_type=='dilate'):
            gloss_shape = gloss_shape / len(trainloader)
            gloss_temporal = gloss_temporal / len(trainloader)
            losses_train.append(gloss.item())
            losses_train_shape.append(gloss_shape.item())
            losses_train_temporal.append(gloss_temporal.item())
        if (loss_type=='mse'):
            losses_train.append(gloss.item())

        if(verbose):
            if (epoch % print_every == 0):
                print('epoch ', epoch, ' loss ',loss.item(),' loss shape ',loss_shape.item(),' loss temporal ',loss_temporal.item())
                u = eval_model(net,testloader, gamma, verbose=0)
                losses_mse.append(u[0])
                losses_dtw.append(u[1])
                losses_tdi.append(u[2])
                print(' Eval mse= ', u[0],' dtw= ', u[1] ,' tdi= ', u[2])
                if loss_type=='mse':
                  scheduler.step(u[0])
                else :
                  dil = u[1]*alpha + u[2]*(1-alpha)
                  scheduler.step(dil)

    print(losses_train)


def eval_model(net,loader, gamma,verbose=1):
    criterion = torch.nn.MSELoss()
    losses_mse = []
    losses_dtw = []
    losses_tdi = []
    net.eval()

    with torch.no_grad():
      for i, data in enumerate(loader, 0):
          loss_mse, loss_dtw, loss_tdi = torch.tensor(0),torch.tensor(0),torch.tensor(0)
          inputs, target = data 
          inputs = inputs.to(device, non_blocking=True)
          target = target.to(device, non_blocking=True)
          batch_size, N_output = target.shape[0:2]
          num_variables = target.shape[2] if len(target.shape) > 2 else 1
          outputs = net(inputs)

          # MSE
          loss_mse = criterion(target,outputs)
          loss_dtw, loss_tdi = 0,0
          # DTW and TDI
          for k in range(batch_size):
            for v in range(num_variables) :
              target_k_cpu = target[k,:,v:v+1].view(-1).detach().cpu().numpy()
              output_k_cpu = outputs[k,:,v:v+1].view(-1).detach().cpu().numpy()

              path, sim = dtw_path(target_k_cpu, output_k_cpu)
              loss_dtw += sim

              Dist = 0
              targ_len = target_k_cpu.shape[0]
              for i,j in path:
                      Dist += (i-j)*(i-j)
              loss_tdi += Dist / (targ_len*targ_len)

          loss_dtw = loss_dtw /(batch_size*num_variables)
          loss_tdi = loss_tdi / (batch_size*num_variables)

          losses_mse.append( loss_mse.item() )
          losses_dtw.append( loss_dtw )
          losses_tdi.append( loss_tdi )
    net.train()
    mean_mse, mean_dtw, mean_tdi = np.array(losses_mse).mean(), np.array(losses_dtw).mean(), np.array(losses_tdi).mean()
    if verbose :
      print(f'\nEvaluation Results - MSE: {mean_mse:.4f}, DTW: {mean_dtw:.4f}, TDI: {mean_tdi:.4f}')

    return mean_mse, mean_dtw, mean_tdi
