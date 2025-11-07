import torch as th
import torch.nn as nn
from torch.nn.functional import relu

#from https://github.com/antspy/inception_v1.pytorch/blob/master/inception_v1.py#L74
class Inception_piece(nn.Module):
    def __init__(self, kernel, out):
        super().__init__()

        self.kernel = kernel
        self.out = out

        #mixed 'name'_bn
        self.conv_bn = nn.LazyConv1d(out_channels=1,        kernel_size=self.kernel, stride=1, padding=self.kernel-1)
        #mixed 'name'
        self.conv    = nn.LazyConv1d(out_channels=self.out, kernel_size=self.kernel, stride=1, padding=0)

    def forward(self,input):
        output = relu(self.conv_bn(input))
        return relu(self.conv(output))

class Inception_variant(nn.Module):
    def __init__(self, depth_dim):
        super().__init__()

        self.depth_dim = depth_dim

        #mixed 'name'_(2,16)_bn
        self.conv_2_16 = Inception_piece(2,16)

        #mixed 'name'_(4,32)_bn
        self.conv_4_32 = Inception_piece(4,32)

        #mixed 'name'_(8,64)_bn
        self.conv_8_64 = Inception_piece(8,64)

        #mixed 'name'_(16,32)_bn
        self.conv_32_16 = Inception_piece(32,16)

        #mixed 'name'_(64,8)_bn
        self.conv_64_8 = Inception_piece(64,8)

    def forward(self, input):

        output1 = relu(self.conv_2_16(input))
        output2 = relu(self.conv_4_32(input))
        output3 = relu(self.conv_8_64(input))
        output4 = relu(self.conv_32_16(input))
        output5 = relu(self.conv_64_8(input))

        c = th.cat((output1, output2, output3, output4, output5), dim = self.depth_dim)
        #return th.transpose(c,-1,-2)
        return c

###

class TransposeLayer(nn.Module):
    def __init__(self, dim1, dim2, verbose = False):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.verbose = verbose
    def forward(self, input):
        if self.verbose:
            print("Before:", input.size())
        inputReshape = th.transpose(input, self.dim1, self.dim2)
        if self.verbose:
            print("After:", inputReshape.size())
        return inputReshape
    
class TransposeWrapper(nn.Module):
    def __init__(self, dim1, dim2, module, verbose = False):
        super().__init__()
        self.module = module
        self.dim1 = dim1
        self.dim2 = dim2
        self.verbose = verbose
    def forward(self, input):
        if self.verbose:
            print("Before:", input.size())
        inputReshape = th.transpose(input, self.dim1, self.dim2)
        if self.verbose:
            print("After:", inputReshape.size())
        y = self.module(inputReshape)
        if self.verbose:
            print("Check:", th.transpose(y, self.dim1, self.dim2).size())
        return th.transpose(y, self.dim1, self.dim2)
    
###

class DebugLayer(nn.Module):
    def __init__(self, msg = ""):
        super().__init__()
        self.msg = msg
    def forward(self, input):
        print(self.msg, input.size())
        return input
###

#From https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        print("1",x.size())

        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        print("2",x_reshape.size())

        y = self.module(x_reshape)


        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

###

#From https://stackoverflow.com/a/64265525
class extract_tensor(nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        #return tensor[:, -1, :]
        return tensor

###