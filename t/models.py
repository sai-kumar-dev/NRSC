import torch.nn as nn   

# PINN Model Definition with LeakyReLU
class OceanHeatFluxPINN(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, output_dim=2, num_layers=4, negative_slope=0.01):
        super(OceanHeatFluxPINN, self).__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LeakyReLU(negative_slope=negative_slope))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope))

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)
