
def visualiser_avec_mcd_loaderformat(loader, out_of_mc, N_input, N_output, which_batch, which_data, variables, reduce_input_window = False) :
  if reduce_input_window :
    index_x, index_y = torch.arange(N_input - N_output, N_input).to(device), torch.arange(N_input, N_output+N_input).to(device)
  else : 
    index_x, index_y = torch.arange(0, N_input).to(device), torch.arange(N_input, N_output+N_input).to(device)
  v = len(variables)
  legende = []
  colorsssss = ["red", "lightcoral", "firebrick", "darksalmon", "rosyblue",
                "cornflowerblue", "lightskyblue", "dodgerblue","mediumslateblue", "mediumpurple", 
                "lightgreen", "turquoise", "green", "lightseagreen" "mediumaquamarine",
                "navajowhite", "gold", "y" "khaki","palegoldenrod"]
  for w in range(v):
    for i, data in enumerate(loader, 0):
      if i!=which_batch:
        pass
      else:
        inputs, target = data
        x = inputs[which_data,:,w].to(device)
        y = target[which_data,:,w].to(device)
        #print(out_of_mc.shape)
        y_pred_mean = out_of_mc[i,which_data,:,w,0].to(device)
        y_pred_ectype = out_of_mc[i,which_data,:,w,1].to(device)
        #print("y_pred_mean", y_pred_mean.shape)
        #print("y_pred_ectype", y_pred_ectype.shape)
    if reduce_input_window : 
      plt.plot(np.asarray(index_x.cpu()), x[-N_output:].cpu(), color = colorsssss[w*5])
    else : 
      plt.plot(np.asarray(index_x.cpu()), x.cpu(), color = colorsssss[w*5])
    plt.plot(np.asarray(index_y.cpu()), y.cpu(), color = colorsssss[w*5+1])
    plt.plot(np.asarray(index_y.cpu()), y_pred_mean.cpu(), color = colorsssss[w*5+2])
    plt.plot(np.asarray(index_y.cpu()), (y_pred_mean+y_pred_ectype).cpu(), color = colorsssss[w*5], linestyle='dashed', lw = 0.8)
    plt.plot(np.asarray(index_y.cpu()), (y_pred_mean-y_pred_ectype).cpu(), color = colorsssss[w*5], linestyle='dashed', lw=0.8)
    #legende = legende.extend([str(variables[w])+': Input', str(variables[w])+': Target',
     #                         str(variables[w])+': Prediction', str(variables[w])+': Incertitude haute',
      #                        str(variables[w])+': Incertitude basse'])
  #plt.legend(legende)
    plt.legend(['Input', 'Target', 'Prediction', 'Confidence (95%)'])#haute', 'Incertitude basse'])
