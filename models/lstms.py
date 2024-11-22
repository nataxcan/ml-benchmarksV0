import math
import torch
from torch import nn
import torch.jit as jit
from torch.nn import Parameter
from typing import List, Tuple, Optional
from torch import Tensor

class LSTM(nn.Module):
    """
    Long short-term memory recurrent unit which has the following update rule:
        it ​= σ(W_xi * ​xt ​+ b_xi ​+ W_hi * ​h(t−1) ​+ b_hi​)
        ft​ = σ(W_xf * ​xt ​+ b_xf ​+ W_hf * ​h(t−1) ​+ b_hf​)
        gt ​= tanh(W_xg * ​xt ​+ b_xg ​+ W_hg * ​h(t−1) ​+ b_hg​)
        ot ​= σ(W_xo * ​xt ​+ b_xo​ + W_ho ​h(t−1) ​+ b_ho​)
        ct ​= ft​ ⊙ c(t−1) ​+ it ​⊙ gt​
        ht ​= ot​ ⊙ tanh(ct​)​
    """
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size

        # LSTM parameters
        # self.weight_xh = nn.Parameter(torch.Tensor(input_size, 4*hidden_size))
        # self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, 4*hidden_size))
        # self.bias_xh = nn.Parameter(torch.Tensor(4*hidden_size))
        # self.bias_hh = nn.Parameter(torch.Tensor(4*hidden_size))
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))

        # Initialize parameters
        self.reset_params()

    def reset_params(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            nn.init.uniform_(param, -std, std)
        # Initialize forget gate bias to 1
        self.bias_ih.data[self.hidden_size:2*self.hidden_size].fill_(1.0)
        self.bias_hh.data[self.hidden_size:2*self.hidden_size].fill_(1.0)

    def forward(self, x):
        """
        Args:
            x: input with shape (N, T, D) where N is number of samples, T is
                number of timestep and D is input size which must be equal to
                self.input_size.

        Returns:
            y: output with a shape of (N, T, H) where H is hidden size
        """

        # Transpose input for efficient vectorized calculation. After transposing
        # the input will have (T, N, D).
        x = x.transpose(0, 1)

        # Unpack dimensions
        T, N, H = x.shape[0], x.shape[1], self.hidden_size

        # Initialize hidden and cell states to zero. There will be one hidden
        # and cell state for each input, so they will have shape of (N, H)
        h0 = torch.zeros(N, H, device=x.device)
        c0 = torch.zeros(N, H, device=x.device)
        
        # Define a list to store outputs. We will then stack them.
        y = []

        ht_1 = h0
        ct_1 = c0
        for t in range(T):
            # LSTM update rule
            # xh = torch.addmm(self.bias_xh, x[t], self.weight_xh) 
            # hh = torch.addmm(self.bias_hh, ht_1, self.weight_hh)
            gates = x[t] @ self.weight_ih.t() + self.bias_ih + ht_1 @ self.weight_hh.t() + self.bias_hh
            it, ft, gt, ot = gates.chunk(4, 1)
            # add_res = xh + hh
            it = torch.sigmoid(it)
            ft = torch.sigmoid(ft)
            gt = torch.tanh(gt)
            ot = torch.sigmoid(ot)
            ct = ft * ct_1 + it * gt
            ht = ot * torch.tanh(ct)

            # Store output
            y.append(ht)

            # For the next iteration c(t-1) and h(t-1) will be current ct and ht
            ct_1 = ct
            ht_1 = ht

        # Stack the outputs. After this operation, output will have shape of
        # (T, N, H)
        y = torch.stack(y)

        # Switch time and batch dimension, (T, N, H) -> (N, T, H)
        y = y.transpose(0, 1)
        return y, None


class LSTMFaster(nn.Module):
    """
    Long short-term memory recurrent unit which has the following update rule:
        it ​= σ(W_xi * ​xt ​+ b_xi ​+ W_hi * ​h(t−1) ​+ b_hi​)
        ft​ = σ(W_xf * ​xt ​+ b_xf ​+ W_hf * ​h(t−1) ​+ b_hf​)
        gt ​= tanh(W_xg * ​xt ​+ b_xg ​+ W_hg * ​h(t−1) ​+ b_hg​)
        ot ​= σ(W_xo * ​xt ​+ b_xo​ + W_ho ​h(t−1) ​+ b_ho​)
        ct ​= ft​ ⊙ c(t−1) ​+ it ​⊙ gt​
        ht ​= ot​ ⊙ tanh(ct​)​
    """
    def __init__(self, input_size, hidden_size):
        super(LSTMFaster, self).__init__()
        '''
        we're gonna compute the x multiplication with the matrix
        in parallel before doing the rest of the operations
        '''

        self.hidden_size = hidden_size

        # LSTM parameters
        # self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        # self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        # self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        # self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.hh_layer = nn.Linear(input_size, 4 * hidden_size)
        self.ih_layer = nn.Linear(input_size, 4 * hidden_size)

        # Initialize parameters
        self.reset_params()

    def reset_params(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            nn.init.uniform_(param, -std, std)
        # Initialize forget gate bias to 1
        # self.bias_ih.data[self.hidden_size:2*self.hidden_size].fill_(1.0)
        # self.bias_hh.data[self.hidden_size:2*self.hidden_size].fill_(1.0)

    def forward(self, x):
        """
        Args:
            x: input with shape (N, T, D) where N is number of samples, T is
                number of timestep and D is input size which must be equal to
                self.input_size.

        Returns:
            y: output with a shape of (N, T, H) where H is hidden size
        """

        # Transpose input for efficient vectorized calculation. After transposing
        # the input will have (T, N, D).
        x = x.transpose(0, 1)

        # Unpack dimensions
        T, N, H = x.shape[0], x.shape[1], self.hidden_size

        # Initialize hidden and cell states to zero. There will be one hidden
        # and cell state for each input, so they will have shape of (N, H)
        h0 = torch.zeros(N, H, device=x.device).type(x.dtype)
        c0 = torch.zeros(N, H, device=x.device).type(x.dtype)
        
        # Define a list to store outputs. We will then stack them.
        y = torch.zeros(T, N, H, device=x.device)

        ht_1 = h0
        ct_1 = c0
        # print('initial x', x.shape)
        xhs = self.ih_layer(x)
        # print("hidden x's ", xhs.shape)
        for t in range(T):
            # LSTM update rule
            # xh = torch.addmm(self.bias_xh, x[t], self.weight_xh)
            # hh = torch.addmm(self.bias_hh, ht_1, self.weight_hh.t())
            hh = self.hh_layer(ht_1)
            # print("hh layer output", hh.shape)
            # gates = x[t] @ self.weight_ih.t() + self.bias_ih + ht_1 @ self.weight_hh.t() + self.bias_hh
            # it, ft, gt, ot = (xh + hh).chunk(4, 1)


            # it, ft, ot, gt = (xhs[t] + hh).chunk(4, 1)
            # it = torch.sigmoid(it)
            # ft = torch.sigmoid(ft)
            # ot = torch.sigmoid(ot)
            # gt = torch.tanh(gt)

            # it, ft, ot, gt = (xhs[t] + hh).chunk(4, 1)
            v = xhs[t] + hh
            # print("the v vector", v.shape)
            g1, gt = torch.sigmoid(v[:,:3*H]), torch.tanh(v[:,3*H:])
            # print("g1 and gt before chunking", g1.shape, gt.shape)
            it, ft, ot = g1.chunk(3, 1)
            # print("the 3 boys", it.shape)

            # print("==>", ft.shape, ct_1.shape, it.shape, gt.shape)
            ct = ft * ct_1 + it * gt
            ht = ot * torch.tanh(ct)

            # Store output
            y[t] = ht

            # For the next iteration c(t-1) and h(t-1) will be current ct and ht
            ct_1 = ct
            ht_1 = ht

        # Switch time and batch dimension, (T, N, H) -> (N, T, H)
        y = y.transpose(0, 1)
        return y, None    

class LSTMUnrolled(nn.Module):
    """
    Long short-term memory recurrent unit which has the following update rule:
        it ​= σ(W_xi * ​xt ​+ b_xi ​+ W_hi * ​h(t−1) ​+ b_hi​)
        ft​ = σ(W_xf * ​xt ​+ b_xf ​+ W_hf * ​h(t−1) ​+ b_hf​)
        gt ​= tanh(W_xg * ​xt ​+ b_xg ​+ W_hg * ​h(t−1) ​+ b_hg​)
        ot ​= σ(W_xo * ​xt ​+ b_xo​ + W_ho ​h(t−1) ​+ b_ho​)
        ct ​= ft​ ⊙ c(t−1) ​+ it ​⊙ gt​
        ht ​= ot​ ⊙ tanh(ct​)​
    """
    def __init__(self, input_size, hidden_size, sequence_length):
        super(LSTMUnrolled, self).__init__()

        self.hidden_size = hidden_size

        # LSTM parameters
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.ih_weights = nn.ParameterList([self.weight_ih.clone() for _ in range(sequence_length)])
        self.hh_weights = nn.ParameterList([self.weight_hh.clone() for _ in range(sequence_length)])
        self.ih_biases = nn.ParameterList([self.bias_ih.clone() for _ in range(sequence_length)])
        self.hh_biases = nn.ParameterList([self.bias_hh.clone() for _ in range(sequence_length)])

        # Initialize parameters
        self.reset_params()

    def reset_params(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            nn.init.uniform_(param, -std, std)
        # Initialize forget gate bias to 1
        self.bias_ih.data[self.hidden_size:2*self.hidden_size].fill_(1.0)
        self.bias_hh.data[self.hidden_size:2*self.hidden_size].fill_(1.0)

    def forward(self, x):
        """
        Args:
            x: input with shape (N, T, D) where N is number of samples, T is
                number of timestep and D is input size which must be equal to
                self.input_size.

        Returns:
            y: output with a shape of (N, T, H) where H is hidden size
        """

        # Transpose input for efficient vectorized calculation. After transposing
        # the input will have (T, N, D).
        x = x.transpose(0, 1)

        # Unpack dimensions
        T, N, H = x.shape[0], x.shape[1], self.hidden_size

        # Initialize hidden and cell states to zero. There will be one hidden
        # and cell state for each input, so they will have shape of (N, H)
        h0 = torch.zeros(N, H, device=x.device).type(x.dtype)
        c0 = torch.zeros(N, H, device=x.device).type(x.dtype)
        
        # Define a list to store outputs. We will then stack them.
        y = []

        ht_1 = h0
        ct_1 = c0
        for t in range(T):
            # LSTM update rule
            # xh = torch.addmm(self.bias_xh, x[t], self.weight_xh) 
            # hh = torch.addmm(self.bias_hh, ht_1, self.weight_hh)
            weight_ih, weight_hh = self.ih_weights[t], self.hh_weights[t]
            bias_ih, bias_hh = self.ih_biases[t], self.hh_biases[t]
            gates = x[t] @ weight_ih + bias_ih + ht_1 @ weight_hh.t() + bias_hh
            it, ft, gt, ot = gates.chunk(4, 1)
            # add_res = xh + hh
            it = torch.sigmoid(it)
            ft = torch.sigmoid(ft)
            gt = torch.tanh(gt)
            ot = torch.sigmoid(ot)
            ct = ft * ct_1 + it * gt
            ht = ot * torch.tanh(ct)

            # Store output
            y.append(ht)

            # For the next iteration c(t-1) and h(t-1) will be current ct and ht
            ct_1 = ct
            ht_1 = ht

        # Stack the outputs. After this operation, output will have shape of
        # (T, N, H)
        y = torch.stack(y)

        # Switch time and batch dimension, (T, N, H) -> (N, T, H)
        y = y.transpose(0, 1)
        return y, None


# ----------------------------------------------------------------------------------------------------------------------
class JitLSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, roll_amount=16):
        super(JitLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        self.roll_amount = roll_amount

        self.reset_parameters()

    # def reset_parameters(self):
    #     # Initialize weights in LSTM style
    #     stdv = 1.0 / math.sqrt(self.hidden_size)
    #     for weight in self.parameters():
    #         weight.data.uniform_(-stdv, stdv)     
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)

        # Initialize weight matrices
        self.weight_ih.data.uniform_(-stdv, stdv)
        self.weight_hh.data.uniform_(-stdv, stdv)
        
        # Initialize biases to zero
        self.bias_ih.data.zero_()
        self.bias_hh.data.zero_()
        
        # Set forget gate biases to 1
        # The gates are ordered as input, forget, cell, output
        # So indices for forget gate are hidden_size to 2*hidden_size
        self.bias_ih.data[self.hidden_size:2*self.hidden_size].fill_(1)
        self.bias_hh.data[self.hidden_size:2*self.hidden_size].fill_(1)
   
    @jit.script_method
    def forward(self, x, hx):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]
        x = x.view(-1, x.size(1))
        h, c = hx  # Unpack the previous hidden and cell states
        gates = (torch.mm(x, self.weight_ih.t()) + self.bias_ih) + (torch.mm(h, self.weight_hh.t()) + self.bias_hh)
        
        # Split the gates
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)
        
        # Activation functions
        input_gate = torch.sigmoid(i_gate)
        forget_gate = torch.sigmoid(f_gate)
        cell_gate = torch.tanh(g_gate)
        output_gate = torch.sigmoid(o_gate)
        
        # Update cell state
        c_next = forget_gate * c + input_gate * cell_gate
        # Compute the hidden state
        h_next = output_gate * torch.tanh(c_next)
        
        # Return the new hidden state and cell state as a tuple
        c_next = torch.roll(c_next, shifts=self.roll_amount, dims=1)
        return h_next, c_next

# ----------------------------------------------------------------------------------------------------------------------
class JitLSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(JitLSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, x, hidden):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = x.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])

        h, c = hidden  # Unpack hidden state and cell state
        for i in range(len(inputs)):
            h, c = self.cell(inputs[i], (h, c))  # Update h and c
            outputs += [h]

        return torch.stack(outputs), (h, c)

# ----------------------------------------------------------------------------------------------------------------------
class JitLSTM(jit.ScriptModule):
    __constants__ = ['hidden_size', 'num_layers', 'batch_first', 'layers']

    def __init__(self, input_size, hidden_size, num_layers, batch_first=False, bias=True, roll_amount=16):
        super(JitLSTM, self).__init__()
        # The following are not implemented.
        assert bias

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        if num_layers == 1:
            self.layers = nn.ModuleList([JitLSTMLayer(JitLSTMCell, input_size, hidden_size)])
        else:
            self.layers = nn.ModuleList(
                [JitLSTMLayer(JitLSTMCell, input_size, hidden_size, roll_amount=roll_amount)] +
                [JitLSTMLayer(JitLSTMCell, hidden_size, hidden_size, roll_amount=roll_amount) for _ in range(num_layers - 1)]
            )

    @jit.script_method
    def forward(self, x, hx=None):
        # type: (Tensor, Optional[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]

        # Handle batch_first cases
        if self.batch_first:
            x = x.permute(1, 0, 2)

        batch_size = x.size(1)

        if hx is None:
            zeros = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            h = zeros  # h_0
            c = zeros  # c_0
        else:
            h, c = hx

        h_n = []
        c_n = []

        output = x
        i = 0
        for rnn_layer in self.layers:
            h_layer = h[i]
            c_layer = c[i]

            output, (h_layer, c_layer) = rnn_layer(output, (h_layer, c_layer))

            h_n.append(h_layer)
            c_n.append(c_layer)
            i += 1

        # Don't forget to handle batch_first cases for the output too!
        if self.batch_first:
            output = output.permute(1, 0, 2)

        h_n_stacked = torch.stack(h_n, dim=0)
        c_n_stacked = torch.stack(c_n, dim=0)

        return output, (h_n_stacked, c_n_stacked)
    

class OptimizedCustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False, device='cpu'):
        super(OptimizedCustomLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.x_layer = nn.Linear(input_size, 4 * hidden_size, device=device)
        self.h_layer = nn.Linear(hidden_size, 4 * hidden_size, device=device)
        
    def forward(self, x):
        if not self.batch_first:
            x = x.transpose(0, 1)
        batch_size, seq_len, _ = x.size()
        
        # if hidden is None:
        #     h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        #     c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        # else:
        #     h_t, c_t = hidden
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        x_gates = self.x_layer(x)
        
        outputs = []
        
        for t in range(seq_len):
            h_gates = self.h_layer(h_t)
            gates = x_gates[:, t, :] + h_gates
            i_t, f_t, g_t, o_t = gates.chunk(4, dim=1)
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            outputs.append(h_t)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs, (h_t, c_t)

class OptimizedCustomLSTM2(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False):
        super(OptimizedCustomLSTM2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.combined_layer = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        
    def forward(self, x):
        if not self.batch_first:
            x = x.transpose(0, 1)
        batch_size, seq_len, _ = x.size()
        
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            combined = torch.cat((x_t, h_t), dim=1)
            gates = self.combined_layer(combined)
            i_t, f_t, g_t, o_t = gates.chunk(4, dim=1)
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            outputs.append(h_t)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs, (h_t, c_t)


class OptimizedCustomLSTM3(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False, device='cpu'):
        super(OptimizedCustomLSTM3, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.combined_layer = nn.Linear(input_size + hidden_size, 4 * hidden_size, device=device)
        
    def forward(self, x, hidden=None):
        if not self.batch_first:
            x = x.transpose(0, 1)
        batch_size, seq_len, _ = x.size()
        
        if hidden is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t, c_t = hidden
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            combined = torch.cat((x_t, h_t), dim=1)
            gates = self.combined_layer(combined)
            i_t, f_t, g_t, o_t = gates.chunk(4, dim=1)
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            outputs.append(h_t)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs, (h_t, c_t)