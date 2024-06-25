import torch.nn as nn

class GenderDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(GenderDiscriminator, self).__init__()
        self.layer1 = nn.Linear(input_dim, 300)
        self.layer2 = nn.Linear(300, 100)
        self.layer3 = nn.Linear(100, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x