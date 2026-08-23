
class LateFusionMLP(torch.nn.Module):
    def __init__(self, one_output_size, num_variables, fc_units, mlp_dropout, num_grulstm_layers):
        super(LateFusionMLP, self).__init__()
        self.one_output_size = one_output_size
        self.num_variables = num_variables
        self.mlp_dropout = mlp_dropout
        self.num_grulstm_layers = num_grulstm_layers
        self.fc_units = fc_units
        self.fc = nn.Linear(self.one_output_size*self.num_variables, self.fc_units)
        self.out = nn.Linear(self.fc_units, self.num_variables*self.one_output_size)

    def forward(self, input):
        flattened_input = input.view(input.shape[0], -1)
        output = F.relu( self.fc(flattened_input) )
        output = F.dropout(output, self.mlp_dropout)
        output = self.out(output)
        output = output.view(input.shape[0], self.one_output_size, self.num_variables)
        return output
