import torch
from torch import nn

'''

all have mixer 25 and 

gru layer basic:
- 1 conv1d per gate (so there's 6)
- a batch norm per recurrent step
v2:
- just removing zeros_like
v4:
- one batch norm
- batch norm run once after all recurrent steps
v3:
- replaces z and r gates with the integer 1
v1:
- z and r gates are dim-1 outputs of the conv1d's
v5:
- all gates are computed with 2 conv1d's that are chunked, and we use concatenation
- just one norm after recurrent steps (compare with v4)
v6:
- same as v5 the refactor
- but then gates are scalar outputs of the conv1d now
v7:
- like v6 but with silu instead of tanh
v8:
- like v7 but with groups in the convolutions
v9:
- same as base pgru

'''

class Flipper(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)

class TrainableHState(nn.Module):
    def __init__(self, dim):
        super(TrainableHState, self).__init__()
        self.h = nn.Parameter(torch.randn(1, dim, 1))

    def forward(self, x):
        b, _, l = x.shape
        return self.h.expand(b, -1, l)

class ParallelGRULayer(nn.Module):
    def __init__(self, num_filters, num_recurrent_steps=3, h_init_mode='trainable', mixer=True, mixer_amount=25,
                 norm='batch'):
        super(ParallelGRULayer, self).__init__()
        self.s2szx = nn.Conv1d(num_filters, num_filters, 1)
        self.s2szh = nn.Conv1d(num_filters, num_filters, 1)
        self.s2srx = nn.Conv1d(num_filters, num_filters, 1)
        self.s2srh = nn.Conv1d(num_filters, num_filters, 1)
        self.s2snx = nn.Conv1d(num_filters, num_filters, 1)
        self.s2snh = nn.Conv1d(num_filters, num_filters, 1)
        self.recurrent_steps = num_recurrent_steps
        self.norms = nn.ModuleList([nn.BatchNorm1d(num_filters) for _ in range(num_recurrent_steps)])
        self.hinit = TrainableHState(num_filters)
        
        self.mixer = nn.Sequential(nn.Conv1d(num_filters, num_filters, mixer_amount, padding=mixer_amount // 2), nn.ReLU())
    
    def forward(self, x):
        h = torch.zeros_like(x)
        h = self.hinit(h)
        for i in range(self.recurrent_steps):
            h = self.mixer(h)
            zt = torch.sigmoid(self.s2szx(x) + self.s2szh(h))
            rt = torch.sigmoid(self.s2srx(x) + self.s2srh(h))
            nt = torch.tanh(self.s2snx(x) + self.s2snh(h * rt))
            h = (1 - zt) * h + zt * nt
            x = x[:,:,1:]
            h = h[:,:,:-1]
            h = self.norms[i](h)
        return h


# removing the zeros_like has a tiny effect on latency and throughput
class ParallelGRULayerv4(nn.Module):
    def __init__(self, num_filters, num_recurrent_steps=3, mixer_amount=25):
        super(ParallelGRULayerv4, self).__init__()
        self.s2szx = nn.Conv1d(num_filters, num_filters, 1)
        self.s2szh = nn.Conv1d(num_filters, num_filters, 1)
        self.s2srx = nn.Conv1d(num_filters, num_filters, 1)
        self.s2srh = nn.Conv1d(num_filters, num_filters, 1)
        self.s2snx = nn.Conv1d(num_filters, num_filters, 1)
        self.s2snh = nn.Conv1d(num_filters, num_filters, 1)
        self.recurrent_steps = num_recurrent_steps
        self.norm = nn.BatchNorm1d(num_filters)
        self.hinit = TrainableHState(num_filters)
        
        self.mixer = nn.Sequential(nn.Conv1d(num_filters, num_filters, mixer_amount, padding=mixer_amount // 2), nn.ReLU())
    
    def forward(self, x):
        # h = torch.zeros_like(x)
        h = self.hinit(x)
        for i in range(self.recurrent_steps):
            h = self.mixer(h)
            zt = torch.sigmoid(self.s2szx(x) + self.s2szh(h))
            rt = torch.sigmoid(self.s2srx(x) + self.s2srh(h))
            nt = torch.tanh(self.s2snx(x) + self.s2snh(h * rt))
            h = (1 - zt) * h + zt * nt
            x = x[:,:,1:]
            h = h[:,:,:-1]
        h = self.norm(h)
        return h
    

# removing the zeros_like has a tiny effect on latency and throughput
class ParallelGRULayerv2(nn.Module):
    def __init__(self, num_filters, num_recurrent_steps=3, mixer_amount=25):
        super(ParallelGRULayerv2, self).__init__()
        self.s2szx = nn.Conv1d(num_filters, num_filters, 1)
        self.s2szh = nn.Conv1d(num_filters, num_filters, 1)
        self.s2srx = nn.Conv1d(num_filters, num_filters, 1)
        self.s2srh = nn.Conv1d(num_filters, num_filters, 1)
        self.s2snx = nn.Conv1d(num_filters, num_filters, 1)
        self.s2snh = nn.Conv1d(num_filters, num_filters, 1)
        self.recurrent_steps = num_recurrent_steps
        self.norms = nn.ModuleList([nn.BatchNorm1d(num_filters) for _ in range(num_recurrent_steps)])
        self.hinit = TrainableHState(num_filters)
        
        self.mixer = nn.Sequential(nn.Conv1d(num_filters, num_filters, mixer_amount, padding=mixer_amount // 2), nn.ReLU())
    
    def forward(self, x):
        # h = torch.zeros_like(x)
        h = self.hinit(x)
        for i in range(self.recurrent_steps):
            h = self.mixer(h)
            zt = torch.sigmoid(self.s2szx(x) + self.s2szh(h))
            rt = torch.sigmoid(self.s2srx(x) + self.s2srh(h))
            nt = torch.tanh(self.s2snx(x) + self.s2snh(h * rt))
            h = (1 - zt) * h + zt * nt
            x = x[:,:,1:]
            h = h[:,:,:-1]
            h = self.norms[i](h)
        return h

# removing z and r gates has a massive effect on throughput, almost doubling it
class ParallelGRULayerv3(nn.Module):
    def __init__(self, num_filters, num_recurrent_steps=3, mixer_amount=25):
        super(ParallelGRULayerv3, self).__init__()
        self.s2szx = nn.Conv1d(num_filters, num_filters, 1)
        self.s2szh = nn.Conv1d(num_filters, num_filters, 1)
        self.s2srx = nn.Conv1d(num_filters, num_filters, 1)
        self.s2srh = nn.Conv1d(num_filters, num_filters, 1)
        self.s2snx = nn.Conv1d(num_filters, num_filters, 1)
        self.s2snh = nn.Conv1d(num_filters, num_filters, 1)
        self.recurrent_steps = num_recurrent_steps
        self.norms = nn.ModuleList([nn.BatchNorm1d(num_filters) for _ in range(num_recurrent_steps)])
        self.hinit = TrainableHState(num_filters)
        
        self.mixer = nn.Sequential(nn.Conv1d(num_filters, num_filters, mixer_amount, padding=mixer_amount // 2), nn.ReLU())
    
    def forward(self, x):
        # h = torch.zeros_like(x)
        h = self.hinit(x)
        for i in range(self.recurrent_steps):
            h = self.mixer(h)
            # zt = torch.sigmoid(self.s2szx(x) + self.s2szh(h))
            # rt = torch.sigmoid(self.s2srx(x) + self.s2srh(h))
            rt, zt = 1, 1
            nt = torch.tanh(self.s2snx(x) + self.s2snh(h * rt))
            h = (1 - zt) * h + zt * nt
            x = x[:,:,1:]
            h = h[:,:,:-1]
            h = self.norms[i](h)
        return h

    
    # you can turn z and r into 1-dim outputs (simple gates) and it'll impact your model a lot
# at small dims, but very little at big dims, it also hurts latency
class ParallelGRULayerv1(nn.Module):
    def __init__(self, num_filters, num_recurrent_steps=3, mixer_amount=25):
        super(ParallelGRULayerv1, self).__init__()
        self.s2szx = nn.Conv1d(num_filters, 1, 1)
        self.s2szh = nn.Conv1d(num_filters, 1, 1)
        self.s2srx = nn.Conv1d(num_filters, 1, 1)
        self.s2srh = nn.Conv1d(num_filters, 1, 1)
        self.s2snx = nn.Conv1d(num_filters, num_filters, 1)
        self.s2snh = nn.Conv1d(num_filters, num_filters, 1)
        self.recurrent_steps = num_recurrent_steps
        self.norms = nn.ModuleList([nn.BatchNorm1d(num_filters) for _ in range(num_recurrent_steps)])
        self.hinit = TrainableHState(num_filters)
        
        self.mixer = nn.Sequential(nn.Conv1d(num_filters, num_filters, mixer_amount, padding=mixer_amount // 2), nn.ReLU())
    
    def forward(self, x):
        # h = torch.zeros_like(x)
        h = self.hinit(x)
        for i in range(self.recurrent_steps):
            h = self.mixer(h)
            zt = torch.sigmoid(self.s2szx(x) + self.s2szh(h))
            rt = torch.sigmoid(self.s2srx(x) + self.s2srh(h))
            nt = torch.tanh(self.s2snx(x) + self.s2snh(h * rt))
            h = (1 - zt) * h + zt * nt
            x = x[:,:,1:]
            h = h[:,:,:-1]
            h = self.norms[i](h)
        return h
    

    # if we fuse all the s2s layers into the bare minimum (2), we get mild increases in
# throughput at small dims, with higher latency at larger dims, and a drop-off of through
# put at highest dim/bs (I think then we're relying more on memory transfers than on flops)
class ParallelGRULayerv5(nn.Module):
    def __init__(self, num_filters, num_recurrent_steps=3, mixer_amount=25):
        super(ParallelGRULayerv5, self).__init__()
        # self.s2szx = nn.Conv1d(num_filters, num_filters, 1)
        # self.s2szh = nn.Conv1d(num_filters, num_filters, 1)
        # self.s2srx = nn.Conv1d(num_filters, num_filters, 1)
        # self.s2srh = nn.Conv1d(num_filters, num_filters, 1)
        # self.s2snx = nn.Conv1d(num_filters, num_filters, 1)
        # self.s2snh = nn.Conv1d(num_filters, num_filters, 1)

        self.s2s1 = nn.Conv1d(num_filters*2, num_filters*2, 1)
        self.s2s2 = nn.Conv1d(num_filters*2, num_filters, 1)

        self.recurrent_steps = num_recurrent_steps
        self.norm = nn.BatchNorm1d(num_filters)
        self.hinit = TrainableHState(num_filters)
        
        self.mixer = nn.Sequential(nn.Conv1d(num_filters, num_filters, mixer_amount, padding=mixer_amount // 2), nn.ReLU())
    
    def forward(self, x):
        # h = torch.zeros_like(x)
        h = self.hinit(x)
        for _ in range(self.recurrent_steps):
            h = self.mixer(h)
            zt, rt = torch.chunk(torch.sigmoid(self.s2s1(torch.cat((x, h), dim=1))), 2, dim=1)
            # zt = torch.sigmoid(self.s2szx(x) + self.s2szh(h))
            # rt = torch.sigmoid(self.s2srx(x) + self.s2srh(h))
            # nt = torch.tanh(self.s2snx(x) + self.s2snh(h * rt))
            nt = torch.tanh(self.s2s2(torch.cat((x, h*rt), dim=1)))
            h = (1 - zt) * h + zt * nt
            x = x[:,:,1:]
            h = h[:,:,:-1]
        h = self.norm(h)
        return h



# if we again turn zt and rt into 1dim outputs we get 1.something speedups
class ParallelGRULayerv6(nn.Module):
    def __init__(self, num_filters, num_recurrent_steps=3, mixer_amount=25):
        super(ParallelGRULayerv6, self).__init__()

        self.s2s1 = nn.Conv1d(num_filters*2, 2, 1)
        self.s2s2 = nn.Conv1d(num_filters*2, num_filters, 1)

        self.recurrent_steps = num_recurrent_steps
        self.norm = nn.BatchNorm1d(num_filters)
        self.hinit = TrainableHState(num_filters)
        
        self.mixer = nn.Sequential(nn.Conv1d(num_filters, num_filters, mixer_amount, padding=mixer_amount // 2), nn.ReLU())
    
    def forward(self, x):
        h = self.hinit(x)
        for _ in range(self.recurrent_steps):
            h = self.mixer(h)
            zt, rt = torch.chunk(torch.sigmoid(self.s2s1(torch.cat((x, h), dim=1))), 2, dim=1)
            nt = torch.tanh(self.s2s2(torch.cat((x, h*rt), dim=1)))
            h = (1 - zt) * h + zt * nt
            x = x[:,:,1:]
            h = h[:,:,:-1]
        h = self.norm(h)
        return h


# if we then replace the tanh with a GELU... we hurt throughput by about 10%
# if we then replace the tanh with a SiLU... it's a little less bad
class ParallelGRULayerv7(nn.Module):
    def __init__(self, num_filters, num_recurrent_steps=3, mixer_amount=25):
        super(ParallelGRULayerv7, self).__init__()

        self.s2s1 = nn.Conv1d(num_filters*2, 2, 1)
        self.s2s2 = nn.Conv1d(num_filters*2, num_filters, 1)

        self.recurrent_steps = num_recurrent_steps
        self.norm = nn.BatchNorm1d(num_filters)
        self.hinit = TrainableHState(num_filters)
        self.act = nn.SiLU()
        self.mixer = nn.Sequential(nn.Conv1d(num_filters, num_filters, mixer_amount, padding=mixer_amount // 2), nn.ReLU())
    
    def forward(self, x):
        h = self.hinit(x)
        for _ in range(self.recurrent_steps):
            h = self.mixer(h)
            zt, rt = torch.chunk(torch.sigmoid(self.s2s1(torch.cat((x, h), dim=1))), 2, dim=1)
            nt = self.act(self.s2s2(torch.cat((x, h*rt), dim=1)))
            h = (1 - zt) * h + zt * nt
            x = x[:,:,1:]
            h = h[:,:,:-1]
        h = self.norm(h)
        return h


# if we then add groups to the convolutions, it doesn't have much of an impact
class ParallelGRULayerv8(nn.Module):
    def __init__(self, num_filters, num_recurrent_steps=3, mixer_amount=25):
        super(ParallelGRULayerv8, self).__init__()

        self.s2s1 = nn.Conv1d(num_filters*2, 2, 1, groups=2)
        self.s2s2 = nn.Conv1d(num_filters*2, num_filters, 1, groups=4)

        self.recurrent_steps = num_recurrent_steps
        self.norm = nn.BatchNorm1d(num_filters)
        self.hinit = TrainableHState(num_filters)
        self.act = nn.SiLU()
        self.mixer = nn.Sequential(nn.Conv1d(num_filters, num_filters, mixer_amount, padding=mixer_amount // 2), nn.ReLU())
    
    def forward(self, x):
        h = self.hinit(x)
        for _ in range(self.recurrent_steps):
            h = self.mixer(h)
            zt, rt = torch.chunk(torch.sigmoid(self.s2s1(torch.cat((x, h), dim=1))), 2, dim=1)
            nt = self.act(self.s2s2(torch.cat((x, h*rt), dim=1)))
            h = (1 - zt) * h + zt * nt
            x = x[:,:,1:]
            h = h[:,:,:-1]
        h = self.norm(h)
        return h

class ParallelGRULayerv9(nn.Module):
    def __init__(self, num_filters, num_recurrent_steps=3, h_init_mode='trainable', mixer=True, mixer_amount=25,
                 norm='batch'):
        super(ParallelGRULayerv9, self).__init__()
        self.s2szx = nn.Conv1d(num_filters, num_filters, 1)
        self.s2szh = nn.Conv1d(num_filters, num_filters, 1)
        self.s2srx = nn.Conv1d(num_filters, num_filters, 1)
        self.s2srh = nn.Conv1d(num_filters, num_filters, 1)
        self.s2snx = nn.Conv1d(num_filters, num_filters, 1)
        self.s2snh = nn.Conv1d(num_filters, num_filters, 1)
        self.recurrent_steps = num_recurrent_steps
        self.norms = nn.ModuleList([nn.BatchNorm1d(num_filters) for _ in range(num_recurrent_steps)])
        self.hinit = TrainableHState(num_filters)
        
        self.mixer = nn.Sequential(nn.Conv1d(num_filters, num_filters, mixer_amount, padding=mixer_amount // 2), nn.ReLU())
    
    def forward(self, x):
        h = torch.zeros_like(x)
        h = self.hinit(h)
        for i in range(self.recurrent_steps):
            h = self.mixer(h)
            zt = torch.sigmoid(self.s2szx(x) + self.s2szh(h))
            rt = torch.sigmoid(self.s2srx(x) + self.s2srh(h))
            nt = torch.tanh(self.s2snx(x) + self.s2snh(h * rt))
            h = (1 - zt) * h + zt * nt
            x = x[:,:,1:]
            h = h[:,:,:-1]
            h = self.norms[i](h)
        return h
