from torch import nn

class RNNClasssifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim

        self.rnn = nn.RNN(
            self.input_dim, 
            self.hidden_dim, 
            self.layer_dim,
            batch_first=True,
            nonlinearity='relu'
            )
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, x):
        # x:   [batch, time_step, input_dim]
        # out: [layer_dim, batch, hidden_dim]
        # h_n: [layer_dim, batch, hidden_dim]
        out, h_n = self.rnn(x, None)
        out = self.fc(out[:, -1, :])
        return out

if __name__ == '__main__':
    from torchsummary import summary
    model = RNNClasssifier(
        input_dim=28,
        hidden_dim=128,
        layer_dim=1,
        output_dim=10
    )
    summary(model, (28, 28), batch_size=1, device='cpu')