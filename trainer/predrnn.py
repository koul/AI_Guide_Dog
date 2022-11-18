import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from trainer.layers.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell
# from config import *

from trainer.tsne import visualization



class SpatioTemporalLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, height, width, kernel_size, bias, conv_dropout):
        """
        Initialize SpatioTemporalLSTMCell cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(SpatioTemporalLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self._forget_bias = 1.0

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=7 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
        #             out_channels=7 * self.hidden_dim,
        #             kernel_size=self.kernel_size,
        #             padding=self.padding,
        #             bias=self.bias),
        #     nn.LayerNorm([7 * self.hidden_dim, self.kernel_size[0], self.kernel_size[1]])
        # )

        # self.conv_dropout = nn.Dropout(p=conv_dropout) if conv_dropout > 0.0 else None
        if conv_dropout > 0:
            print(f'Applying Conv Dropout of {conv_dropout} to cell')
            self.conv_dropout = nn.Dropout(p=conv_dropout)
        else:
            self.conv_dropout = None

        self.layernorm = nn.LayerNorm([7 * self.hidden_dim, height, width])

        self.conv_memory_to_o = nn.Conv2d(in_channels=self.hidden_dim*2,
                                        out_channels=self.hidden_dim,
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        bias=False)

        self.conv_memory_to_h_next = nn.Conv2d(in_channels=self.hidden_dim*2,
                                            out_channels=self.hidden_dim,
                                            kernel_size=1,
                                            padding=0,
                                            bias=False)

    def forward(self, input_tensor, cur_state):
        """
        Propagate SpatioTemporalLSTMCell cell.
        Parameters
        ----------
        input_tensor: batch of images at the current frame (b, c, h, w)
        cur_state: (h_cur, c_cur, m_cur) h_cur, c_cur, m_cur: (b, hidden_dim, h, w)

        Returns
        ----------
        h_next: hidden_out (b, hidden_dim, h, w)
        c_next: c_memory out (b, hidden_dim, h, w)
        m_next: m_memory out (b, hidden_dim, h, w)
        delta_c: change in c_memory (b, hidden_dim, h, w)
        delta_m: change in m_memory (b, hidden_dim, h, w)
        """
        h_cur, c_cur, m_cur = cur_state
        # print(f'input_tensor: {input_tensor.shape}, h_cur: {h_cur.shape}')
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        # print(f'combined: {combined.shape}')
        combined_conv = self.conv(combined)

        if self.conv_dropout is not None:  # Added conv dropout
            combined_conv = self.conv_dropout(combined_conv)

        normalized_conv = self.layernorm(combined_conv)
        cc_i, cc_f, cc_g, cc_i_prime, cc_f_prime, cc_g_prime, cc_o = torch.split(normalized_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f + self._forget_bias )
        g = torch.tanh(cc_g)
        # o = torch.sigmoid(cc_o)

        i_prime = torch.sigmoid(cc_i_prime)
        f_prime = torch.sigmoid(cc_f_prime + self._forget_bias)
        g_prime = torch.tanh(cc_g_prime)

        delta_c = i * g
        c_next = f * c_cur + i * g

        delta_m = i_prime * g_prime
        m_next = f_prime * m_cur + delta_m

        c_m_next = torch.cat((c_next, m_next), dim=1)
        o_with_mem = torch.sigmoid(cc_o + self.conv_memory_to_o(c_m_next))
        # print(f'o_with_mem: {o_with_mem.shape}')
        h_next = o_with_mem * torch.tanh(self.conv_memory_to_h_next(c_m_next))
        # print(f'h_next: {h_next.shape}')

        return h_next, c_next, m_next, delta_c, delta_m

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class PredRNN(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, height, width, kernel_size, num_layers,
                 batch_first=False, bias=True, conv_dropout=0.0, return_all_layers=False):
        super(PredRNN, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        # print(f'hidden_dims: {hidden_dim}, {num_layers}')
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        # self.return_all_layers = return_all_layers
        if return_all_layers:
            print("Warning: return_all_layers not supported by PredRnn")

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(SpatioTemporalLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          height=height,
                                          width=width,
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,
                                          conv_dropout=conv_dropout))

        self.cell_list = nn.ModuleList(cell_list)
        ### Predrnn extra attrs ###
        self.mem_decouple_loss = nn.MSELoss()

        self.conv_last = nn.Conv2d(hidden_dim[-1], hidden_dim[-1], 1, stride=1, padding=0, bias=False) # decouple weights
        self.decouple_conv = nn.Conv2d(hidden_dim[0], hidden_dim[0], 1, stride=1, padding=0, bias=False) # decouple weights

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()
        # print(f'Batch size {b}')

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        time_output_list = []
        # last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        # Outputs across all layers
        all_h_t = []
        all_c_t = []
        all_delta_c = []
        all_delta_m = []
        decouple_losses = []

        ### init inputs for first timestep at each layer ###
        for i in range(self.num_layers):
            zero_matrix = torch.zeros([b, self.hidden_dim[i], h, w]).to(self.device) ## NOTE: Possible incompatible device error here
            all_h_t.append(zero_matrix)
            all_c_t.append(zero_matrix)
            all_delta_c.append(zero_matrix)
            all_delta_m.append(zero_matrix)

        memory = torch.zeros([b, self.hidden_dim[0], h, w]).to(self.device) ## NOTE: Possible incompatible device error here


        for t in range(seq_len):
            all_h_t[0], all_c_t[0], memory, delta_c, delta_m = self.cell_list[0](input_tensor=input_tensor[:, t, :, :, :],
                                                                     cur_state=[all_h_t[0], all_c_t[0], memory])

            all_delta_c[0] = F.normalize(self.decouple_conv(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2) # normalize across h x w
            all_delta_m[0] = F.normalize(self.decouple_conv(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            for l in range(1, self.num_layers):
                all_h_t[l], all_c_t[l], memory, delta_c, delta_m = self.cell_list[l](input_tensor=all_h_t[l-1],
                                                                                cur_state=[all_h_t[l], all_c_t[l], memory])
                all_delta_c[i] = F.normalize(self.decouple_conv(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                all_delta_m[i] = F.normalize(self.decouple_conv(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            # print(f'all_h_t[self.num_layers - 1]: {all_h_t[self.num_layers - 1].shape}')
            all_h_t[self.num_layers - 1] = self.conv_last(all_h_t[self.num_layers - 1])
            # print(f'h_t last: {all_h_t[self.num_layers - 1].shape}')
            # print(f'h_t last: {len(all_h_t)}')
            time_output_list.append(all_h_t[self.num_layers - 1])

            for i in range(0, self.num_layers):
                decouple_losses.append(
                    torch.mean(torch.abs(torch.cosine_similarity(all_delta_c[i], all_delta_m[i], dim=2))))

        decouple_loss = torch.mean(torch.stack(decouple_losses, dim=0))

        time_outputs = torch.stack(time_output_list, dim=1)
        # print(f'time_outputs: {time_outputs.shape}, {time_output_list[0].shape} {len(time_output_list)}')

        # for layer_idx in range(self.num_layers):

        #     h, c = hidden_state[layer_idx]
        #     output_inner = []
        #     for t in range(seq_len):
        #         h, c, m, delta_c, delta_m = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
        #                                          cur_state=[h, c])
        #         output_inner.append(h) #[batch_size, self.hidden_dim, height, width]

        #     layer_output = torch.stack(output_inner, dim=1) #[batch_size,t,self.hidden_dim, height, width]
        #     cur_layer_input = layer_output

        #     layer_output_list.append(layer_output)
        #     last_state_list.append([h, c])

        # if not self.return_all_layers:
        #     layer_output_list = layer_output_list[-1:]
        #     last_state_list = last_state_list[-1:]

        return time_outputs, decouple_loss  # time_outputs: (b, t, c, h, w)

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class PredRnnModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, height, width,
                 batch_first=False, bias=True, conv_dropout=0.0, return_all_layers=False, num_classes = 3):
        super(PredRnnModel, self).__init__()
        self.predRNN = PredRNN(input_dim, hidden_dim, height, width, kernel_size, num_layers,batch_first, bias, conv_dropout, return_all_layers)
        self.linear = nn.Linear(hidden_dim * height * width, num_classes)

    def forward(self, input_tensor, hidden_state=None):
      x, decouple_loss = self.predRNN(input_tensor)
      x = torch.flatten(x[:,-1,:,:,:], start_dim=1)

      # print(x.shape)  	# torch.Size([2, 8, 8388608])
      x = self.linear(x) #op: [batch, num_classes]
      # print(x.shape)
      return x, decouple_loss
