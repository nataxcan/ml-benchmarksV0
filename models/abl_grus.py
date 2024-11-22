
import torch
import torch.jit as jit
import torch.nn as nn
from torch.nn import Parameter
from typing import List, Tuple, Optional
from torch import Tensor
import math


# ----------------------------------------------------------------------------------------------------------------------
class JitGRUCellNoz(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(JitGRUCellNoz, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.Tensor(2 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(2 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(2 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(2 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    @jit.script_method
    def forward(self, x, hidden):
        # type: (Tensor, Tensor) -> Tensor
        x = x.view(-1, x.size(1))
        x_results = torch.mm(x, self.weight_ih.t()) + self.bias_ih
        h_results = torch.mm(hidden, self.weight_hh.t()) + self.bias_hh

        i_r, i_n = x_results.chunk(2, 1)
        h_r, h_n = h_results.chunk(2, 1)

        r = torch.sigmoid(i_r + h_r)
        n = torch.tanh(i_n + r * h_n)

        return n


# ----------------------------------------------------------------------------------------------------------------------
class JitGRUCellNor(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(JitGRUCellNor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.Tensor(2 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(2 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(2 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(2 * hidden_size))
        self.bias_n = Parameter(torch.Tensor(hidden_size))
        self.bias_z = Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    @jit.script_method
    def forward(self, x, hidden):
        # type: (Tensor, Tensor) -> Tensor
        x = x.view(-1, x.size(1))
        x_results = torch.mm(x, self.weight_ih.t()) + self.bias_ih
        h_results = torch.mm(hidden, self.weight_hh.t()) + self.bias_hh

        i_z, i_n = x_results.chunk(2, 1)
        h_z, h_n = h_results.chunk(2, 1)

        z = torch.sigmoid(i_z + h_z + self.bias_z)
        n = torch.tanh(i_n + h_n + self.bias_n)

        return n - torch.mul(n, z) + torch.mul(z, hidden)

# ----------------------------------------------------------------------------------------------------------------------
class JitGRUCellNozr(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(JitGRUCellNozr, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(hidden_size))
        self.bias_hh = Parameter(torch.Tensor(hidden_size))
        self.bias_n = Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    @jit.script_method
    def forward(self, x, hidden):
        # type: (Tensor, Tensor) -> Tensor
        x = x.view(-1, x.size(1))
        i_n = torch.mm(x, self.weight_ih.t()) + self.bias_ih
        h_n = torch.mm(hidden, self.weight_hh.t()) + self.bias_hh
        n = torch.tanh(i_n + h_n + self.bias_n)

        return n


# ----------------------------------------------------------------------------------------------------------------------
class JitGRUCellOnezr(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(JitGRUCellOnezr, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.Tensor(hidden_size + 2, input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size + 2, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(hidden_size + 2))
        self.bias_hh = Parameter(torch.Tensor(hidden_size + 2))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    @jit.script_method
    def forward(self, x, hidden):
        # type: (Tensor, Tensor) -> Tensor
        x = x.view(-1, x.size(1))
        x_results = torch.mm(x, self.weight_ih.t()) + self.bias_ih
        h_results = torch.mm(hidden, self.weight_hh.t()) + self.bias_hh
        i_r, i_z, i_n = x_results[:,:self.hidden_size], x_results[:,-2:-1], x_results[:,-1:]
        h_r, h_z, h_n = h_results[:,:self.hidden_size], h_results[:,-2:-1], h_results[:,-1:]

        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)

        return n - torch.mul(n, z) + torch.mul(z, hidden)

# ----------------------------------------------------------------------------------------------------------------------
class JitGRULayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(JitGRULayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, x, hidden):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        inputs = x.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])

        for i in range(len(inputs)):
            hidden = self.cell(inputs[i], hidden)
            outputs += [hidden]

        return torch.stack(outputs), hidden

# ----------------------------------------------------------------------------------------------------------------------
class JitGRUAblations(jit.ScriptModule):
    __constants__ = ['hidden_size', 'num_layers', 'batch_first', 'layers']

    def __init__(self, input_size, hidden_size, num_layers, cell_class, batch_first=False, bias=True):
        super(JitGRUAblations, self).__init__()
        # The following are not implemented.
        assert bias

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        if num_layers == 1:
            self.layers = nn.ModuleList([JitGRULayer(cell_class, input_size, hidden_size)])
        else:
            self.layers = nn.ModuleList([JitGRULayer(cell_class, input_size, hidden_size)] + [JitGRULayer(cell_class, hidden_size, hidden_size)
                                                                                              for _ in range(num_layers - 1)])

    @jit.script_method
    def forward(self, x, h=None):
        # type: (Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        output_states = jit.annotate(List[Tensor], [])

        # Handle batch_first cases
        if self.batch_first:
            x = x.permute(1, 0, 2)

        if h is None:
            h = torch.zeros(self.num_layers, x.shape[1], self.hidden_size, dtype=x.dtype, device=x.device)

        output = x
        i = 0

        for rnn_layer in self.layers:
            output, hidden = rnn_layer(output, h[i])
            output_states += [hidden]
            i += 1

        # Don't forget to handle batch_first cases for the output too!
        if self.batch_first:
            output = output.permute(1, 0, 2)

        return output, torch.stack(output_states)

from functools import partial
def get_jitgru_ablations(ablname, input_size, hidden_size):
    if ablname == 'noz':
        return JitGRUAblations(input_size, hidden_size, 1, JitGRUCellNoz, batch_first=True)
    if ablname == 'nor':
        return JitGRUAblations(input_size, hidden_size, 1, JitGRUCellNor, batch_first=True)
    if ablname == 'nozr':
        return JitGRUAblations(input_size, hidden_size, 1, JitGRUCellNozr, batch_first=True)
    if ablname == 'onezr':
        return JitGRUAblations(input_size, hidden_size, 1, JitGRUCellOnezr, batch_first=True)


if __name__ == '__main__':
    import torch

    # Set input dimensions and hidden size
    input_size = 10
    hidden_size = 20
    sequence_length = 5
    batch_size = 3

    # Create random input data and initial hidden state
    x = torch.randn(batch_size, sequence_length, input_size)
    h = torch.randn(1, batch_size, hidden_size)

    # Instantiate the models
    model_noz = get_jitgru_ablations('noz', input_size, hidden_size)
    model_nor = get_jitgru_ablations('nor', input_size, hidden_size)
    model_nozr = get_jitgru_ablations('nozr', input_size, hidden_size)
    model_onezr = get_jitgru_ablations('onezr', input_size, hidden_size)

    # Run a forward pass
    output_noz, hidden_noz = model_noz(x, h)
    output_nor, hidden_nor = model_nor(x, h)
    output_nozr, hidden_nozr = model_nozr(x, h)
    output_onezr, hidden_onezr = model_onezr(x, h)

    print(f"Output NOZ: {output_noz.shape}, Hidden NOZ: {hidden_noz.shape}")
    print(f"Output NOR: {output_nor.shape}, Hidden NOR: {hidden_nor.shape}")
    print(f"Output NOZR: {output_nozr.shape}, Hidden NOZR: {hidden_nozr.shape}")
    print(f"Output ONEZR: {output_onezr.shape}, Hidden ONEZR: {hidden_onezr.shape}")