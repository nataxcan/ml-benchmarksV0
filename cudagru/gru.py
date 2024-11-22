import math
from torch import nn
from torch.autograd import Function
import torch

import gru_cuda

torch.manual_seed(42)

class GRUFunction(Function):
    @staticmethod
    def forward(ctx, input, x2h_w, h2h_w, x2h_b, h2h_b, old_h):
        x = input.view(-1, input.size(1)).contiguous()
        outputs = gru_cuda.forward(x, x2h_w, h2h_w, x2h_b, h2h_b, old_h)
        new_h = outputs[0]
        variables = outputs[1:] + [old_h, x, x2h_w, h2h_w]
        ctx.save_for_backward(*variables)

        return new_h

    @staticmethod
    def backward(ctx, grad_hy):
        grad_input_weights, grad_hidden_weights, grad_input_bias, grad_hidden_bias, grad_hx = gru_cuda.backward(
            grad_hy.contiguous(), *ctx.saved_variables
        )

        return None, grad_input_weights, grad_hidden_weights, grad_input_bias, grad_hidden_bias, grad_hx


class GRUCell(nn.Module):
    def __init__(self, input_features, state_size):
        super(GRUCell, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.x2h_weights = nn.Parameter(torch.Tensor(3 * state_size, input_features))
        self.h2h_weights = nn.Parameter(torch.Tensor(3 * state_size, state_size))
        self.x2h_bias = nn.Parameter(torch.Tensor(1, 3 * state_size))
        self.h2h_bias = nn.Parameter(torch.Tensor(1, 3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.state_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, state):
        input = input.view(-1, input.size(1))

        return GRUFunction.apply(
            input, 
            self.x2h_weights, self.h2h_weights,
            self.x2h_bias, self.h2h_bias,
            state    
        )


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, bidirectional=False):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        
        # Create GRU cells for each layer
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            for direction in range(1 + int(bidirectional)):
                layer_input_size = input_size if layer == 0 else hidden_size * (1 + int(bidirectional))
                self.cells.append(GRUCell(layer_input_size, hidden_size))
        
    def forward(self, input, hx=None):
        # Ensure input is of shape [seq_len, batch, input_size]
        if self.batch_first:
            input = input.transpose(0, 1)
        
        seq_len, batch_size, _ = input.size()
        num_directions = 2 if self.bidirectional else 1
        
        if hx is None:
            hx = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, 
                             dtype=input.dtype, device=input.device)
        
        # Reshape hx to (num_layers * num_directions, batch_size, hidden_size)
        hx = hx.view(self.num_layers, num_directions, batch_size, self.hidden_size)
        
        output = []
        
        for layer in range(self.num_layers):
            layer_output = []
            for direction in range(num_directions):
                idx = layer * num_directions + direction
                layer_hx = hx[layer, direction]
                
                if direction == 0:
                    seq_iter = range(seq_len)
                else:
                    seq_iter = range(seq_len - 1, -1, -1)
                
                for t in seq_iter:
                    layer_input = input[t] if layer == 0 else layer_output[-1][:, :self.hidden_size] if direction == 0 else layer_output[-1][:, self.hidden_size:]
                    layer_hx = self.cells[idx](layer_input, layer_hx)
                    layer_output.append(layer_hx)
            
            layer_output = torch.stack(layer_output, dim=0)
            if direction == 1:
                layer_output = layer_output.flip(0)
            
            if self.bidirectional:
                layer_output = torch.cat([layer_output, layer_output.flip(0)], dim=2)
            
            input = layer_output
            output.append(layer_output)
        
        output = output[-1]  # Use the output from the last layer
        
        if self.batch_first:
            output = output.transpose(0, 1)
        
        # Reshape hx back to (num_layers * num_directions, batch_size, hidden_size)
        hx = hx.transpose(0, 1).contiguous().view(-1, batch_size, self.hidden_size)
        
        return output, hx