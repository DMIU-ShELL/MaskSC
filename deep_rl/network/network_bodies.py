#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *

class NatureConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class CTgraphConvBody(nn.Module):
    def __init__(self, in_channels=1):
        super(CTgraphConvBody, self).__init__()
        self.feature_dim = 16
        self.conv1 = layer_init(nn.Conv2d(in_channels, 4, kernel_size=5, stride=1))
        self.conv2 = layer_init(nn.Conv2d(4, 8, kernel_size=3, stride=1))
        self.conv3 = layer_init(nn.Conv2d(8, 16, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(4 * 4 * 16, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class MNISTConvBody(nn.Module):
    def __init__(self, in_channels=1, noisy_linear=False):
        super(MNISTConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2))
        #self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        if noisy_linear:
            self.fc4 = NoisyLinear(6 * 6 * 64, self.feature_dim)
        else:
            self.fc4 = layer_init(nn.Linear(6 * 6 * 64, self.feature_dim))
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        if self.noisy_linear:
            self.fc4.reset_noise()

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        #y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class DDPGConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(DDPGConvBody, self).__init__()
        self.feature_dim = 39 * 39 * 32
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

    def forward(self, x):
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.view(y.size(0), -1)
        return y

class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim, ) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x

class FCBody_CL(nn.Module): # fcbody for continual learning setup
    def __init__(self, state_dim, task_label_dim=None, hidden_units=(64, 64), gate=F.relu):
        super(FCBody_CL, self).__init__()
        if task_label_dim is None:
            dims = (state_dim, ) + hidden_units
        else:
            dims = (state_dim + task_label_dim, ) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]
        self.task_label_dim = task_label_dim

    def forward(self, x, task_label=None, return_layer_output=False, prefix=''):
        if self.task_label_dim is not None:
            assert task_label is not None, '`task_label` should be set'
            x = torch.cat([x, task_label], dim=1)
        #if task_label is not None: x = torch.cat([x, task_label], dim=1)
       
        ret_act = []
        if return_layer_output:
            for i, layer in enumerate(self.layers):
                x = self.gate(layer(x))
                ret_act.append(('{0}.layers.{1}'.format(prefix, i), x))
        else:
            for layer in self.layers:
                x = self.gate(layer(x))
        return x, ret_act

from ..mask_modules.mmn.mask_nets import MultitaskMaskLinear, MultitaskMaskConv2D
from ..mask_modules.mmn.mask_nets import NEW_MASK_RANDOM
from ..mask_modules.mmn.mask_nets import NEW_MASK_LINEAR_COMB
class FCBody_SS(nn.Module): # fcbody for supermask superposition continual learning algorithm
    def __init__(self, state_dim, task_label_dim=None, hidden_units=(64, 64), gate=F.relu, discrete_mask=True, num_tasks=3, new_task_mask=NEW_MASK_RANDOM):
        super(FCBody_SS, self).__init__()
        if task_label_dim is None:
            dims = (state_dim, ) + hidden_units
        else:
            dims = (state_dim + task_label_dim, ) + hidden_units
        self.layers = nn.ModuleList([MultitaskMaskLinear(dim_in, dim_out, discrete=discrete_mask, \
            num_tasks=num_tasks, new_mask_type=new_task_mask) \
            for dim_in, dim_out in zip(dims[:-1], dims[1:])
        ])
        self.gate = gate
        self.feature_dim = dims[-1]
        self.task_label_dim = task_label_dim

    def forward(self, x, task_label=None, return_layer_output=False, prefix=''):
        if self.task_label_dim is not None:
            assert task_label is not None, '`task_label` should be set'
            x = torch.cat([x, task_label], dim=1)
        #if task_label is not None: x = torch.cat([x, task_label], dim=1)
       
        ret_act = []
        if return_layer_output:
            for i, layer in enumerate(self.layers):
                x = self.gate(layer(x))
                ret_act.append(('{0}.layers.{1}'.format(prefix, i), x))
        else:
            for layer in self.layers:
                x = self.gate(layer(x))
        return x, ret_act


class LayerNormFCBody_CL(nn.Module):
    def __init__(self, state_dim, task_label_dim=None, hidden_units=(256, 256, 256, 256),
                 negative_slope=0.2):
        super(LayerNormFCBody_CL, self).__init__()
        if task_label_dim is None:
            dims = (state_dim,) + hidden_units
        else:
            dims = (state_dim + task_label_dim,) + hidden_units
        self.layers = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])]
        )
        self.first_layer_norm = nn.LayerNorm(dims[1]) if len(dims) > 1 else None
        self.feature_dim = dims[-1]
        self.task_label_dim = task_label_dim
        self.negative_slope = negative_slope

    def _activate(self, x, idx):
        if idx == 0:
            if self.first_layer_norm is not None:
                x = self.first_layer_norm(x)
            return torch.tanh(x)
        return F.leaky_relu(x, negative_slope=self.negative_slope)

    def forward(self, x, task_label=None, return_layer_output=False, prefix=''):
        if self.task_label_dim is not None:
            assert task_label is not None, '`task_label` should be set'
            x = torch.cat([x, task_label], dim=1)

        ret_act = []
        for i, layer in enumerate(self.layers):
            x = self._activate(layer(x), i)
            if return_layer_output:
                ret_act.append((f'{prefix}.layers.{i}', x))
        return x, ret_act


class LayerNormFCBody_SS(nn.Module):
    def __init__(self, state_dim, task_label_dim=None, hidden_units=(256, 256, 256, 256),
                 negative_slope=0.2, discrete_mask=True, num_tasks=3,
                 new_task_mask=NEW_MASK_RANDOM):
        super(LayerNormFCBody_SS, self).__init__()
        if task_label_dim is None:
            dims = (state_dim,) + hidden_units
        else:
            dims = (state_dim + task_label_dim,) + hidden_units
        self.layers = nn.ModuleList([
            MultitaskMaskLinear(
                dim_in,
                dim_out,
                discrete=discrete_mask,
                num_tasks=num_tasks,
                new_mask_type=new_task_mask,
            )
            for dim_in, dim_out in zip(dims[:-1], dims[1:])
        ])
        self.first_layer_norm = nn.LayerNorm(dims[1]) if len(dims) > 1 else None
        self.feature_dim = dims[-1]
        self.task_label_dim = task_label_dim
        self.negative_slope = negative_slope

    def _activate(self, x, idx):
        if idx == 0:
            if self.first_layer_norm is not None:
                x = self.first_layer_norm(x)
            return torch.tanh(x)
        return F.leaky_relu(x, negative_slope=self.negative_slope)

    def forward(self, x, task_label=None, return_layer_output=False, prefix=''):
        if self.task_label_dim is not None:
            assert task_label is not None, '`task_label` should be set'
            x = torch.cat([x, task_label], dim=1)

        ret_act = []
        for i, layer in enumerate(self.layers):
            x = self._activate(layer(x), i)
            if return_layer_output:
                ret_act.append((f'{prefix}.layers.{i}', x))
        return x, ret_act

class TwoLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=(64, 64), gate=F.relu):
        super(TwoLayerFCBodyWithAction, self).__init__()
        hidden_size1, hidden_size2 = hidden_units
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_size1))
        self.fc2 = layer_init(nn.Linear(hidden_size1 + action_dim, hidden_size2))
        self.gate = gate
        self.feature_dim = hidden_size2

    def forward(self, x, action):
        x = self.gate(self.fc1(x))
        phi = self.gate(self.fc2(torch.cat([x, action], dim=1)))
        return phi

class OneLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units, gate=F.relu):
        super(OneLayerFCBodyWithAction, self).__init__()
        self.fc_s = layer_init(nn.Linear(state_dim, hidden_units))
        self.fc_a = layer_init(nn.Linear(action_dim, hidden_units))
        self.gate = gate
        self.feature_dim = hidden_units * 2

    def forward(self, x, action):
        phi = self.gate(torch.cat([self.fc_s(x), self.fc_a(action)], dim=1))
        return phi

class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x

class DummyBody_CL(nn.Module):
    def __init__(self, state_dim, task_label_dim=None):
        super(DummyBody_CL, self).__init__()
        self.feature_dim = state_dim + (0 if task_label_dim is None else task_label_dim)
        self.task_label_dim = task_label_dim

    def forward(self, x, task_label=None, return_layer_output=False, prefix=''):
        if self.task_label_dim is not None:
            assert task_label is not None, '`task_label` should be set'
            x = torch.cat([x, task_label], dim=1)
        return x, []

class DummyBody_CL_Mask(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody_CL_Mask, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x, task_label=None, return_layer_output=False, prefix='', mask=None):
        return x, []

class ConvBody_SS_Modified(nn.Module): # conv body for supermask lifelong learning algorithm
    def __init__(self, state_dim, kernels=[(3,3), (3,3)], strides=[1,1], paddings=[1,1], feature_dim=512, task_label_dim=None, gate=F.relu, discrete_mask=True, num_tasks=3, new_task_mask=NEW_MASK_RANDOM, seed=1):
        super(ConvBody_SS_Modified, self).__init__()

        assert len(state_dim) == 3, 'expected image observations shaped like (H, W, C)'

        def _pair(v):
            return v if isinstance(v, tuple) else (v, v)

        def _conv_out_dim(size, kernel, stride, padding):
            return ((size + 2 * padding - kernel) // stride) + 1

        height, width, in_channels = state_dim
        self.in_channels = in_channels
        self.conv1 = MultitaskMaskConv2D(
            in_channels,
            16,
            kernel_size=kernels[0],
            stride=strides[0],
            padding=paddings[0],
            discrete=discrete_mask,
            num_tasks=num_tasks,
            new_mask_type=new_task_mask,
        )
        self.conv2 = MultitaskMaskConv2D(
            16,
            32,
            kernel_size=kernels[1],
            stride=strides[1],
            padding=paddings[1],
            discrete=discrete_mask,
            num_tasks=num_tasks,
            new_mask_type=new_task_mask,
        )
        
        '''if task_label_dim is None: dims = (state_dim[0], ) + hidden_units
        else: dims = (state_dim[0] + task_label_dim, ) + hidden_units
        self.layers = nn.ModuleList(
            [
                MultitaskMaskConv2D(dim_in, dim_out, kernel_size=kernel, stride=stride, padding=padding, discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask, seed=seed) \
                for dim_in, dim_out, kernel, stride, padding in zip(dims[:-1], dims[1:], kernels, strides, paddings)
            ]
        )

        flattened_in = 128 * max(state_dim) * min(state_dim)
        self.layers.append(MultitaskMaskLinear(flattened_in, feature_dim, num_tasks=num_tasks, new_mask_type=new_task_mask, seed=seed))

        print(f'Network: {self.layers}')'''

        #self.direction_emb = nn.Embedding(4, 4)
        #self.mission_emb = nn.Embedding(100, 16)
        #self.lstm = nn.LSTM(input_size=32 * 7 * 7 + 4 + 16, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        #self.maxp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layer for output
        self.flatten = nn.Flatten()
        kernel1_h, kernel1_w = _pair(kernels[0])
        stride1_h, stride1_w = _pair(strides[0])
        pad1_h, pad1_w = _pair(paddings[0])
        kernel2_h, kernel2_w = _pair(kernels[1])
        stride2_h, stride2_w = _pair(strides[1])
        pad2_h, pad2_w = _pair(paddings[1])

        conv1_h = _conv_out_dim(height, kernel1_h, stride1_h, pad1_h)
        conv1_w = _conv_out_dim(width, kernel1_w, stride1_w, pad1_w)
        conv2_h = _conv_out_dim(conv1_h, kernel2_h, stride2_h, pad2_h)
        conv2_w = _conv_out_dim(conv1_w, kernel2_w, stride2_w, pad2_w)
        flattened_in = 32 * conv2_h * conv2_w
        if task_label_dim is not None:
            flattened_in += task_label_dim

        self.fc = MultitaskMaskLinear(
            flattened_in,
            feature_dim,
            num_tasks=num_tasks,
            new_mask_type=new_task_mask,
        )

        self.gate = gate
        self.feature_dim = feature_dim
        self.task_label_dim = task_label_dim

    def forward(self, x, task_label=None, return_layer_output=False, prefix=''):
        if self.task_label_dim is not None:
            assert task_label is not None, '`task_label` should be set'

        ret_act = []

        # MiniGrid observations arrive as NHWC; Conv2d expects NCHW.
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError(f'expected 4D image batch, got shape {tuple(x.shape)}')
        if x.shape[-1] == self.in_channels:
            x = x.permute(0, 3, 1, 2).contiguous()
        elif x.shape[1] != self.in_channels:
            raise ValueError(
                f'expected {self.in_channels} input channels, got tensor with shape {tuple(x.shape)}'
            )

        # conv1
        y = self.gate(self.conv1(x))
        if return_layer_output:
            ret_act.append(('{0}.conv.1'.format(prefix), y.detach().cpu().reshape(-1,)))
        
        # maxp1
        #y = self.maxp1(y)
        
        # conv2
        y = self.gate(self.conv2(y))
        #print(y.shape)
        if return_layer_output:
            ret_act.append(('{0}.conv.2'.format(prefix), y.detach().cpu().reshape(-1,)))

        # flatten
        y = self.flatten(y)
        #y = y.view(y.shape[0], -1)
        #print(y.shape)
        if self.task_label_dim is not None:
            y = torch.cat([y, task_label], dim=1)
        
        # fc1
        y = self.gate(self.fc(y))
        if return_layer_output:
            ret_act.append(('{0}.fc.1'.format(prefix), y.detach().cpu().reshape(-1,)))
        return y, ret_act

    '''def forward(self, x, task_label=None, return_layer_output=False, prefix=''):
        if self.task_label_dim is not None:
            assert task_label is not None, '`task_label` should be set'
            x = torch.cat([x, task_label], dim=1)
        #if task_label is not None: x = torch.cat([x, task_label], dim=1)
       
        ret_act = []
        if return_layer_output:
            for i, layer in enumerate(self.layers):
                x = self.gate(layer(x))
                ret_act.append(('{0}.layers.{1}'.format(prefix, i), x))
        else:
            for i, layer in enumerate(self.layers):
                x = self.gate(layer(x))

        return x, ret_act'''
