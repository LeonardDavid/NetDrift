import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import numpy as np
import random

class Quantize(Function):
    @staticmethod
    def forward(ctx, input, quantization):
        output = input.clone().detach()
        output = quantization.applyQuantization(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

quantize = Quantize.apply


class ErrorModel(Function):
    @staticmethod
    def forward(ctx, input, index_offset, block_size=64, error_model=None):
        output = input.clone().detach()
        # print(index_offset)
        output = error_model.applyErrorModel(output, index_offset, block_size)
        # print(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None

apply_error_model = ErrorModel.apply


# add for compatibility to every apply_error_model parameters that do not use index_offset and block_size
index_offset_default = np.zeros([2,2])
block_size_default = 1.0

def check_quantization(quantize_train, quantize_eval, training):
    condition = ((quantize_train == True) and (training == True)) or ((quantize_eval == True) and (training == False)) or ((quantize_train == True) and (quantize_eval == True))

    if (condition == True):
        return True
    else:
        return False


class QuantizedActivation(nn.Module):
    def __init__(self, *args, **kwargs):
        self.name = "QuantizedActivation"
        self.layerNR = kwargs.pop('layerNr', None)
        self.quantization = kwargs.pop('quantization', None)
        self.error_model = kwargs.pop('error_model', None)
        self.quantize_train = kwargs.pop('quantize_train', True)
        self.quantize_eval = kwargs.pop('quantize_eval', True)
        self.training = None
        super(QuantizedActivation, self).__init__(*args, **kwargs)

    def forward(self, input):
        output = None
        check_q = check_quantization(self.quantize_train,
         self.quantize_eval, self.training)
        if (check_q == True):
            output = quantize(input, self.quantization)
        else:
            output = input
        if self.error_model is not None:
            output = apply_error_model(output, index_offset_default, block_size_default, self.error_model)
        return output


class QuantizedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        self.name = "QuantizedLinear"
        self.layerNR = kwargs.pop('layerNr', None)
        self.quantization = kwargs.pop('quantization', None)
        self.error_model = kwargs.pop('error_model', None)
        self.quantize_train = kwargs.pop('quantize_train', True)
        self.quantize_eval = kwargs.pop('quantize_eval', True)
        self.an_sim = kwargs.pop('an_sim', None)
        self.array_size = kwargs.pop('array_size', None)
        self.mapping = kwargs.pop('mac_mapping', None)
        self.mapping_distr = kwargs.pop('mac_mapping_distr', None)
        self.sorted_mapping_idx = kwargs.pop('sorted_mac_mapping_idx', None)
        self.performance_mode = kwargs.pop('performance_mode', None)
        self.training = kwargs.pop('train_model', None)
        self.extract_absfreq = kwargs.pop('extract_absfreq', None)
        if self.extract_absfreq is not None:
            self.absfreq = torch.zeros(self.array_size+1, dtype=int).cuda()
        self.test_rtm = kwargs.pop('test_rtm', False)
        self.index_offset = kwargs.pop('index_offset', None)
        self.block_size = kwargs.pop('block_size', None)
        self.protectLayers = kwargs.pop('protectLayers', None)
        self.err_shifts = kwargs.pop('err_shifts', None)
        super(QuantizedLinear, self).__init__(*args, **kwargs)

    def forward(self, input):
        if self.bias is None:
            quantized_weight = None
            check_q = check_quantization(self.quantize_train, self.quantize_eval, self.training)

            if (check_q == True):
                quantized_weight = quantize(self.weight, self.quantization)
            else:
                quantized_weight = self.weight

            if self.error_model is not None:
                if self.test_rtm is not None and self.protectLayers[self.layerNR-1]==0:
                    # print("Linear", self.layerNR)
                    # print(self.block_size)
                    # print("")
                    # print(np.sum(self.index_offset))

                    # nr_elem=0
                    # print(quantized_weight.shape[0])
                    # print(quantized_weight.shape[1])
                    # for i in range(0, quantized_weight.shape[0]):
                    #     for j in range(0, quantized_weight.shape[1]):
                    #         nr_elem += 1
                    #         # print(quantized_weight[i][j])
                    #     # print("\n")
                    # print(nr_elem)

                    # print(self.index_offset.shape[0])
                    # print(self.index_offset.shape[1])

                    err_shift = 0   # number of error shifts
                    shift = 0       # number of shifts (used for reading)
                    for i in range(0, self.index_offset.shape[0]):      #
                        for j in range(0, self.index_offset.shape[1]):  #
                            # start at 1 because AP is on the first element at the beginning, no shift is needed for reading the first value
                            for k in range(1, self.block_size):         #
                                shift += 1
                                if(random.uniform(0.0, 1.0) < self.error_model.p):
                                    err_shift += 1
                                    # 50/50 random possibility (uniform distribution) of right or left err_shift
                                    if(random.choice([-1,1]) == 1):
                                        # right err_shift
                                        if (self.index_offset[i][j] < self.block_size/2): # +1
                                            self.index_offset[i][j] += 1
                                        # self.index_offset[i][j] += 1
                                        # if (self.index_offset[i][j] > self.block_size/2): # +1
                                        #     self.lost_vals_r[i][j] += 1
                                        #     quantized_weight[i][(j+1)*self.block_size - int(self.lost_vals_r[i][j])] = random.choice([-1,1])
                                        # if (self.lost_vals_l[i][j] > 0):
                                        #     self.lost_vals_l[i][j] -= 1
                                    else:
                                        # left err_shift
                                        if (self.index_offset[i][j] < self.block_size/2): # -1
                                            self.index_offset[i][j] -= 1
                                        # self.index_offset[i][j] -= 1
                                        # if(-self.index_offset[i][j] > self.block_size/2): # +1
                                        #     self.lost_vals_l[i][j] += 1
                                        #     quantized_weight[i][j*self.block_size + int(self.lost_vals_l[i][j]) - 1] = random.choice([-1,1])
                                        # if(self.lost_vals_r[i][j] > 0):
                                        #     self.lost_vals_r[i][j] -= 1

                    self.err_shifts[self.layerNR-1] += err_shift

                    # print("local err_shifts: " + str(err_shift) + "/" + str(shift))
                    # print(self.err_shifts)

                    # print(np.sum(self.index_offset))
                    # print(self.index_offset)
                                        
                quantized_weight = ErrorModel.apply(quantized_weight, self.index_offset, self.block_size, self.error_model)

                output = F.linear(input, quantized_weight)
            return output
        else:
            quantized_weight = None
            quantized_bias = None
            check_q = check_quantization(self.quantize_train,
             self.quantize_eval, self.training)
            if (check_q == True):
                quantized_weight = quantize(self.weight, self.quantization)
                quantized_bias = quantize(self.bias, self.quantization)
            else:
                quantized_weight = self.weight
                quantized_bias = self.bias
            if self.error_model is not None:
                quantized_weight = apply_error_model(quantized_weight, index_offset_default, block_size_default, self.error_model)
                quantized_bias = apply_error_model(quantized_bias, index_offset_default, block_size_default, self.error_model)
            return F.linear(input, quantized_weight, quantized_bias)


class QuantizedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.name = "QuantizedConv2d"
        self.layerNR = kwargs.pop('layerNr', None)
        self.quantization = kwargs.pop('quantization', None)
        self.error_model = kwargs.pop('error_model', None)
        self.quantize_train = kwargs.pop('quantize_train', True)
        self.quantize_eval = kwargs.pop('quantize_eval', True)
        self.an_sim = kwargs.pop('an_sim', None)
        self.array_size = kwargs.pop('array_size', None)
        self.mapping = kwargs.pop('mac_mapping', None)
        self.mapping_distr = kwargs.pop('mac_mapping_distr', None)
        self.sorted_mapping_idx = kwargs.pop('sorted_mac_mapping_idx', None)
        self.performance_mode = kwargs.pop('performance_mode', None)
        self.training = kwargs.pop('train_model', None)
        self.extract_absfreq = kwargs.pop('extract_absfreq', None)
        if self.extract_absfreq is not None:
            self.absfreq = torch.zeros(self.array_size+1, dtype=int).cuda()
        self.test_rtm = kwargs.pop('test_rtm', False)
        self.index_offset = kwargs.pop('index_offset', None)
        self.block_size = kwargs.pop('block_size', None)
        self.protectLayers = kwargs.pop('protectLayers', None)
        self.err_shifts = kwargs.pop('err_shifts', None)
        super(QuantizedConv2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        if self.bias is None:
            quantized_weight = None
            check_q = check_quantization(self.quantize_train,
             self.quantize_eval, self.training)
            if (check_q == True):
                quantized_weight = quantize(self.weight, self.quantization)
                # weight_b1 = quantized_weight.view(self.out_channels,-1).cuda()
                # wm = weight_b1.shape
                # input_b1 = F.unfold(input, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride).cuda()
                # im = input_b1.shape
                # print("wm, im", wm, im)
            else:
                quantized_weight = self.weight
                quantized_bias = self.bias
            if self.error_model is not None:

                if self.test_rtm is not None and self.protectLayers[self.layerNR-1]==0:
                    # print("Convolution2D", self.layerNR)
                    # print(self.block_size)
                    # print("")
                    # print(np.sum(self.index_offset))

                    # nr_elem=0
                    # print(quantized_weight.shape[0])
                    # print(quantized_weight.shape[1])
                    # for i in range(0, quantized_weight.shape[0]):
                    #     for j in range(0, quantized_weight.shape[1]):
                    #         nr_elem += 1
                    #         # print(quantized_weight[i][j])
                    #     # print("\n")
                    # print(nr_elem)

                    # print(self.index_offset.shape[0])
                    # print(self.index_offset.shape[1])

                    err_shift = 0   # number of error shifts
                    shift = 0       # number of shifts (used for reading)
                    # iterate over all blocks (row-wise -> swap for loops for column-wise)
                    for i in range(0, self.index_offset.shape[0]):      # 
                        for j in range(0, self.index_offset.shape[1]):  # 
                            # now, read every value from the block
                            # start at 1 because AP is on the first element at the beginning, no shift is needed for reading the first value
                            for k in range(1, self.block_size):         # 
                                shift += 1
                                if(random.uniform(0.0, 1.0) < self.error_model.p):
                                    err_shift += 1
                                    # 50/50 random possibility of right or left err_shift
                                    if(random.choice([-1,1]) == 1):
                                        # right err_shift
                                        if (self.index_offset[i][j] < self.block_size/2): # +1
                                            self.index_offset[i][j] += 1
                                        # self.index_offset[i][j] += 1
                                        # if (self.index_offset[i][j] > self.block_size/2): # +1
                                        #     self.lost_vals_r[i][j] += 1
                                        #     quantized_weight[i][(j+1)*self.block_size - int(self.lost_vals_r[i][j])] = random.choice([-1,1])
                                        # if (self.lost_vals_l[i][j] > 0):
                                        #     self.lost_vals_l[i][j] -= 1
                                    else:
                                        # left err_shift
                                        if (self.index_offset[i][j] < self.block_size/2): # -1
                                            self.index_offset[i][j] -= 1
                                        # self.index_offset[i][j] -= 1
                                        # if(-self.index_offset[i][j] > self.block_size/2): # +1
                                        #     self.lost_vals_l[i][j] += 1
                                        #     quantized_weight[i][j*self.block_size + int(self.lost_vals_l[i][j]) - 1] = random.choice([-1,1])
                                        # if(self.lost_vals_r[i][j] > 0):
                                        #     self.lost_vals_r[i][j] -= 1

                    self.err_shifts[self.layerNR-1] += err_shift

                    # print("local err_shifts: " + str(err_shift) + "/" + str(shift))
                    # print(self.err_shifts)

                    # print(np.sum(self.index_offset))
                    # print(self.index_offset)

                quantized_weight = apply_error_model(quantized_weight, self.index_offset, self.block_size, self.error_model)

                output = F.conv2d(input, quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return output
        else:
            quantized_weight = None
            quantized_bias = None
            check_q = check_quantization(self.quantize_train,
             self.quantize_eval, self.training)
            # check quantization case
            if (check_q == True):
                quantized_weight = quantize(self.weight, self.quantization)
                quantized_bias = quantize(self.bias, self.quantization)
            else:
                quantized_weight = self.weight
                quantized_bias = self.bias
            # check whether error model needs to be applied
            if self.error_model is not None:
                quantized_weight = apply_error_model(quantized_weight, index_offset_default, block_size_default, self.error_model)
                quantized_bias = apply_error_model(quantized_bias, index_offset_default, block_size_default, self.error_model)
            # compute regular 2d conv
            output = F.conv2d(input, quantized_weight, quantized_bias, self.stride,
                              self.padding, self.dilation, self.groups)
            return output
