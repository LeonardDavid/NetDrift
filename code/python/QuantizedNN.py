import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import numpy as np
import random

import metrics.count_len.count_len_endlen as endlen
import metrics.binomial_revert.binomial_revert as bin_revert

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

def read_data(filepath):
    """
    Reads a list of integers from a file and converts it to a PyTorch tensor.

    Args:
        filepath: Path to the file containing the data.

    Returns:
        A PyTorch tensor with the same structure as the data in the file.
    """
    with open(filepath, 'r') as f:
        data = eval(f.read())  # Assuming the data is valid python expression

    # Convert lists to tensors and combine them
    inner_tensors = []
    for outer_list in data:
        inner_tensors.append(outer_list)  

    return torch.tensor(inner_tensors)


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

### read from file parameteres ### 

nr_flip = 1
edge_flag = False 
bitlen = "endlen1"
n_l_r = 1

folder = "q_out_fmnist3x3_endlen"
# folder = "q_out_fmnist5x5_endlen"
# folder = "q_out_fmnist7x7_endlen"

# folder = "q_out_cifar3x3_endlen"

###

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
        self.err_shifts_ind = kwargs.pop('err_shifts_ind', None)
        self.bitflips = kwargs.pop('bitflips', None)
        self.global_bitflip_budget = kwargs.pop('global_bitflip_budget', None)
        self.local_bitflip_budget = kwargs.pop('local_bitflip_budget', None)
        self.nr_run = 1
        self.q_weight = None
        super(QuantizedLinear, self).__init__(*args, **kwargs)

    def forward(self, input):
        if self.bias is None:
            quantized_weight = None
            check_q = check_quantization(self.quantize_train, self.quantize_eval, self.training)

            if (check_q == True):
                quantized_weight = quantize(self.weight, self.quantization)
            else:
                quantized_weight = self.weight

            # if self.protectLayers[self.layerNR-1]==0:
            #     list_of_integers = quantized_weight.cpu().tolist()

            #     try:
            #         with open('qweights_orig_'+str(self.layerNR)+'.txt', 'w') as f:
            #             f.write("[")

            #             # Write the list of integers to the file
            #             for integer in list_of_integers[:-1]:
            #                 f.write(str(integer) + ',\n')

            #             f.write(str(list_of_integers[-1]) + "]")
            #     except FileExistsError:
            #         print("orig already exists")

            if self.error_model is not None:
                if self.test_rtm is not None and self.protectLayers[self.layerNR-1]==0:
                    print("Linear", self.layerNR)
                    # print(self.nr_run)

                    ### read weight tensor from file

                    # # file = "qweights/64/qweights_0.1/qweights_append_3.txt"
                    # # file = "qweights/64/qweights_0.1/qweights_shift1_4.txt"

                    # # file = "q/qweights_shift1_4_mod.txt"
                    # # file = "q/qweights_append_4_lce_mod.txt"

                    # file = "metrics/count_len/q_in/qweights_orig_"+str(self.layerNR)+".txt"
                    # file = "metrics/count_len/q_out_indiv/qweights_orig_"+str(self.layerNR)+"_flip_"+str(n_l_r)+"_"+str(n_l_r)+".txt"
                    # file = "metrics/count_len/q_out/qweights_orig_"+str(self.layerNR)+"_"+str(nr_flip)+"flip"+str(bitlen)+"_"+str(n_l_r)+"_"+str(n_l_r)+".txt"
                    
                    ### ENDLEN ###

                    # if "endlen" in bitlen:
                    #     file = "metrics/count_len/"+str(folder)+"/qweights_orig_"+str(self.layerNR)+"_"+str(nr_flip)+"flip_"+str(bitlen)+".txt"
                    # else:
                    #     if edge_flag:
                    #         file = "metrics/count_len/"+str(folder)+"/qweights_orig_"+str(self.layerNR)+"_"+str(nr_flip)+"flip"+str(bitlen)+"e_"+str(n_l_r)+"_"+str(n_l_r)+".txt"
                    #     else:
                    #         file = "metrics/count_len/"+str(folder)+"/qweights_orig_"+str(self.layerNR)+"_"+str(nr_flip)+"flip"+str(bitlen)+"_"+str(n_l_r)+"_"+str(n_l_r)+".txt"
                    # print(file)
                    # data_tensor = read_data(file).cuda()

                    # # print(data_tensor)
                    # print(data_tensor.shape) 
                    # # L3: [2048, 3136]
                    # # L4: [10, 2048]

                    # quantized_weight = quantize(data_tensor, self.quantization)

                    ### ENDLEN ###

                    quantized_weight_init = quantized_weight

                    err_shift = 0   # number of error shifts
                    shift = 0       # number of shifts (used for reading)
                    for i in range(0, self.index_offset.shape[0]):      #
                        for j in range(0, self.index_offset.shape[1]):  #
                            # # self.index_offset[i][j] += 1
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
                    self.err_shifts_ind[self.layerNR-1].append(err_shift)
                    # print(self.err_shifts_ind)

                    # print("local err_shifts: " + str(err_shift) + "/" + str(shift))
                    # print(self.err_shifts)

                    # print(np.sum(self.index_offset))
                    # print(self.index_offset)
                    if self.nr_run==1:
                        with open("ind_off/ind_off_"+str(self.layerNR)+"_init.txt", "w") as f:
                            for i in range(0, self.index_offset.shape[0]):      # 
                                for j in range(0, self.index_offset.shape[1]):  #
                                    f.write(str(self.index_offset[i][j]) + " ")
                                f.write("\n")


                    ### BINOMIAL CUTOFF ###

                    # reset some index offset values above a certain threshold
                    # quite theoretical as well, because this would mean that error correction is applied only to some blocks, but in practice it is either full ecc or no ecc
                    # some possible thresholds: cut 80% of the amount of values starting from the middle (0, 1, -1, 2, -2 etc) and leave 20% on the edges
                    # or possibility 2: cut 80% of the total sizes starting from the edges (40% on the right, 40% on the left)
                    # significant overhead to be reckoned with, only for counting (and creating histogram)

                    before = np.sum(abs(self.index_offset))

                    # for i in range(0, self.index_offset.shape[0]):      # 
                    #     for j in range(0, self.index_offset.shape[1]):  # 
                    #         if abs(self.index_offset[i][j]) >= 2:
                    #             self.index_offset[i][j] = 0

                    # # 80/20 from middle (total elements)
                    self.index_offset = bin_revert.revert_elements_2d_mid_separate(self.index_offset)
                    # # 80/20 from edges (total bins)
                    # self.index_offset = bin_revert.revert_elements_2d_edges_separate(self.index_offset)

                    after = np.sum(abs(self.index_offset))
                    diff = before-after
                    print(f"{diff} / {diff/before*100}")

                    with open("ind_off/ind_off_"+str(self.layerNR)+"_run"+str(self.nr_run)+".txt", "w") as f:
                        for i in range(0, self.index_offset.shape[0]):      # 
                            for j in range(0, self.index_offset.shape[1]):  #
                                f.write(str(self.index_offset[i][j]) + " ")
                            f.write("\n")

                    ### BINOMIAL CUTOFF ###


                    ### ODD2EVEN ###

                    # perform theoretical shift in case of an odd number of shifts (to help repair alternating structures):
                    # this shifting should also include its own shift errors with the same probability
                    # -> we leave it out for now, just for theoretical testing
                    # -> in future, we could create a best-case and worst-case, in which latter would be that shift error happens also during this "correction"
                    # this would add some overhead in practice

                    # for i in range(0, self.index_offset.shape[0]):      # 
                    #     for j in range(0, self.index_offset.shape[1]):  # 
                    #         if self.index_offset[i][j] % 2 != 0:
                    #             self.index_offset[i][j] += np.sign(self.index_offset[i][j])

                    ### ODD2EVEN ###


                    ### EVEN2ODD ###

                    # perform theoretical shift in case of an odd number of shifts (to help repair alternating structures):
                    # this shifting should also include its own shift errors with the same probability
                    # -> we leave it out for now, just for theoretical testing
                    # -> in future, we could create a best-case and worst-case, in which latter would be that shift error happens also during this "correction"
                    # this would add some overhead in practice

                    # for i in range(0, self.index_offset.shape[0]):      # 
                    #     for j in range(0, self.index_offset.shape[1]):  # 
                    #         if self.index_offset[i][j] != 0 and self.index_offset[i][j] % 2 == 0:
                    #             self.index_offset[i][j] -= np.sign(self.index_offset[i][j])

                    # print(self.index_offset)

                    ### EVEN2ODD ###


                    ### AT RUNTIME ###

                    # # print(quantized_weight)
                    # endlen.apply_1flip(array_type="1D", block_size=self.block_size, data=quantized_weight)
                    # print("endlen flip applied")
                    # # print(quantized_weight)

                    # # print(quantized_weight)
                    # if self.nr_run == 1:
                    #     endlen.apply_1flip_ind_off(array_type="1D", block_size=self.block_size, data=quantized_weight, index_offset=self.index_offset, global_bitflip_budget=self.global_bitflip_budget, local_bitflip_budget=self.local_bitflip_budget)
                    #     print("endlen flip according to index_offset applied")
                    #     self.q_weight = quantized_weight
                    # else:
                    #     quantized_weight = self.q_weight
                    # # print(quantized_weight)

                    ### AT RUNTIME ###


                    self.nr_run += 1


                quantized_weight = ErrorModel.apply(quantized_weight, self.index_offset, self.block_size, self.error_model)


                if self.protectLayers[self.layerNR-1]==0:
                    differences = np.count_nonzero(quantized_weight_init.cpu() != quantized_weight.cpu())
                    self.bitflips[self.layerNR-1].append(differences)
                
                # if self.protectLayers[self.layerNR-1]==0:
                #     list_of_integers = quantized_weight.cpu().tolist()

                #     # Open a file in write mode
                #     with open('qweights_shift1_'+str(self.layerNR)+'.txt', 'w') as f:
                #         f.write("[")

                #         # Write the list of integers to the file
                #         for integer in list_of_integers[:-1]:
                #             f.write(str(integer) + ',\n')

                #         f.write(str(list_of_integers[-1]) + "]")

                # if self.protectLayers[self.layerNR-1]==0:
                #     list_of_integers = quantized_weight.cpu().tolist()
                #     try:
                #         with open('qweights/'+str(self.block_size)+'/qweights_'+str(self.error_model.p)+'/qweights_shift1_'+str(self.layerNR)+'.txt', 'x') as f:
                #             f.write("[")
                #             # Write the list of integers to the file
                #             for integer in list_of_integers[:-1]:
                #                 f.write(str(integer) + ',\n')
                #             f.write(str(list_of_integers[-1]) + "]")
                #             print("Wrote content to shift1 file")
                #     except FileExistsError:
                #         print("shift1 already exists, writing to shift2 file.")
                #         try:
                #             with open('qweights/'+str(self.block_size)+'/qweights_'+str(self.error_model.p)+'/qweights_shift2_'+str(self.layerNR)+'.txt', 'x') as f:
                #                 f.write("[")
                #                 # Write the list of integers to the file
                #                 for integer in list_of_integers[:-1]:
                #                     f.write(str(integer) + ',\n')
                #                 f.write(str(list_of_integers[-1]) + "]")
                #                 print("Wrote content to shift2 file")
                #         except FileExistsError:
                #             print("shift2 already exists, skipping write.")
                        
                #     try:
                #         with open('qweights/'+str(self.block_size)+'/qweights_'+str(self.error_model.p)+'/qweights_shift10_'+str(self.layerNR)+'.txt', 'w') as f:
                #             f.write("[")
                #             # Write the list of integers to the file
                #             for integer in list_of_integers[:-1]:
                #                 f.write(str(integer) + ',\n')
                #             f.write(str(list_of_integers[-1]) + "]")
                #             print("Wrote content to shift10 file")
                #     except FileExistsError:
                #         print("shift10 already exists, skipping write.")

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
        self.err_shifts_ind = kwargs.pop('err_shifts_ind', None)
        self.bitflips = kwargs.pop('bitflips', None)
        self.global_bitflip_budget = kwargs.pop('global_bitflip_budget', None)
        self.local_bitflip_budget = kwargs.pop('local_bitflip_budget', None)
        self.nr_run = 1
        self.q_weight = None
        super(QuantizedConv2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        if self.bias is None:
            quantized_weight = None
            check_q = check_quantization(self.quantize_train, self.quantize_eval, self.training)
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

            # if self.protectLayers[self.layerNR-1]==0:
            #     list_of_integers = quantized_weight.cpu().tolist()

            #     try:
            #         with open('qweights_orig_'+str(self.layerNR)+'.txt', 'w') as f:
            #             f.write("[")

            #             # Write the list of integers to the file
            #             for integer in list_of_integers[:-1]:
            #                 f.write(str(integer) + ',\n')

            #             f.write(str(list_of_integers[-1]) + "]")
            #     except FileExistsError:
            #         print("orig already exists")

            ###


            # if self.protectLayers[self.layerNR-1]==0:
            #     list_of_integers = quantized_weight.cpu().tolist()
            #     try:
            #         with open('qweights_initial0_'+str(self.layerNR)+'.txt', 'x') as f:
            #             # Write the list of integers to the file
            #             for integer in list_of_integers:
            #                 f.write(str(integer) + '\n')
            #             print("Wrote content to initial file")
            #     except FileExistsError:
            #         print("initial already exists, writing to after1 file.")
            #         try:
            #             with open('qweights_after1_'+str(self.layerNR)+'.txt', 'x') as f:
            #                 # Write the list of integers to the file
            #                 for integer in list_of_integers:
            #                     f.write(str(integer) + '\n')
            #                 print("Wrote content to after1 file")
            #         except FileExistsError:
            #             print("after1 exists, skipping write.")


            if self.error_model is not None:

                if self.test_rtm is not None and self.protectLayers[self.layerNR-1]==0:
                    print("Convolution2D", self.layerNR)
                    # print(self.nr_run)

                    # print(quantized_weight.size())
                    # print(self.block_size)
                    # print("")
                    # print(np.sum(self.index_offset))
                    # print(self.index_offset.shape[0])
                    # print(self.index_offset.shape[1])

                    ### read weight tensor from file

                    # # file = "qweights/64/qweights_0.1/qweights_append_1.txt"
                    # # file = "qweights/64/qweights_0.1/qweights_shift1_2.txt"
                    # # file = "q/qweights_shift1_1.txt"

                    # # file = "q/qweights_shift1_2_mod.txt"
                    # # file = "q_index_offset/qweights_shift1_2_mod.txt"

                    # file = "metrics/count_len/q_in/qweights_orig_"+str(self.layerNR)+".txt"
                    # file = "metrics/count_len/q_out_indiv/qweights_orig_"+str(self.layerNR)+"_flip_"+str(n_l_r)+"_"+str(n_l_r)+".txt"
                    # file = "metrics/count_len/q_out/qweights_orig_"+str(self.layerNR)+"_"+str(nr_flip)+"flip"+str(bitlen)+"_"+str(n_l_r)+"_"+str(n_l_r)+".txt"
                    
                    ### ENDLEN ###

                    # if "endlen" in bitlen:
                    #     file = "metrics/count_len/"+str(folder)+"/qweights_orig_"+str(self.layerNR)+"_"+str(nr_flip)+"flip_"+str(bitlen)+".txt"
                    # else:
                    #     if edge_flag:
                    #         file = "metrics/count_len/"+str(folder)+"/qweights_orig_"+str(self.layerNR)+"_"+str(nr_flip)+"flip"+str(bitlen)+"e_"+str(n_l_r)+"_"+str(n_l_r)+".txt"
                    #     else:
                    #         file = "metrics/count_len/"+str(folder)+"/qweights_orig_"+str(self.layerNR)+"_"+str(nr_flip)+"flip"+str(bitlen)+"_"+str(n_l_r)+"_"+str(n_l_r)+".txt"
                    # print(file)
                    # data_tensor = read_data(file).cuda()

                    # # print(data_tensor)
                    # print(data_tensor.shape)
                    # # L1: [64, 1, 3, 3]
                    # # L2: [64, 64, 3, 3]

                    # quantized_weight = quantize(data_tensor, self.quantization)

                    ### ENDLEN ###

                    quantized_weight_init = quantized_weight

                    err_shift = 0   # number of error shifts
                    shift = 0       # number of shifts (used for reading)
                    # iterate over all blocks (row-wise -> swap for loops for column-wise)
                    for i in range(0, self.index_offset.shape[0]):      # 
                        for j in range(0, self.index_offset.shape[1]):  # 
                            # # self.index_offset[i][j] += 1
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
                    self.err_shifts_ind[self.layerNR-1].append(err_shift)
                    # print(self.err_shifts_ind)

                    # print("local err_shifts: " + str(err_shift) + "/" + str(shift))
                    # print(self.err_shifts)

                    # print(np.sum(self.index_offset))
                    # print(self.index_offset)
                    if self.nr_run==1:
                        with open("ind_off/ind_off_"+str(self.layerNR)+"_init.txt", "w") as f:
                            for i in range(0, self.index_offset.shape[0]):      # 
                                for j in range(0, self.index_offset.shape[1]):  #
                                    f.write(str(self.index_offset[i][j]) + " ")
                                f.write("\n")


                    ### BINOMIAL CUTOFF ###

                    # reset some index offset values above a certain threshold
                    # quite theoretical as well, because this would mean that error correction is applied only to some blocks, but in practice it is either full ecc or no ecc
                    # some possible thresholds: cut 80% of the amount of values starting from the middle (0, 1, -1, 2, -2 etc) and leave 20% on the edges
                    # or possibility 2: cut 80% of the total sizes starting from the edges (40% on the right, 40% on the left)
                    # significant overhead to be reckoned with, only for counting (and creating histogram)

                    before = np.sum(abs(self.index_offset))

                    # for i in range(0, self.index_offset.shape[0]):      # 
                    #     for j in range(0, self.index_offset.shape[1]):  # 
                    #         if abs(self.index_offset[i][j]) <= 2:
                    #             self.index_offset[i][j] = 0
                    
                    # # 80/20 from middle (total elements)
                    self.index_offset = bin_revert.revert_elements_2d_mid_separate(self.index_offset)
                    # # 80/20 from edges (total bins)
                    # self.index_offset = bin_revert.revert_elements_2d_edges_separate(self.index_offset)

                    after = np.sum(abs(self.index_offset))
                    # print(f"{before} - {after}")
                    diff = before-after
                    print(f"{diff} / {diff/before*100}")

                    with open("ind_off/ind_off_"+str(self.layerNR)+"_run"+str(self.nr_run)+".txt", "w") as f:
                        for i in range(0, self.index_offset.shape[0]):      # 
                            for j in range(0, self.index_offset.shape[1]):  #
                                f.write(str(self.index_offset[i][j]) + " ")
                            f.write("\n")

                    ### BINOMIAL CUTOFF ###


                    ### ODD2EVEN ###

                    # perform theoretical shift in case of an odd number of shifts (to help repair alternating structures):
                    # this shifting should also include its own shift errors with the same probability
                    # -> we leave it out for now, just for theoretical testing
                    # -> in future, we could create a best-case and worst-case, in which latter would be that shift error happens also during this "correction"
                    # this would add some overhead in practice

                    # for i in range(0, self.index_offset.shape[0]):      # 
                    #     for j in range(0, self.index_offset.shape[1]):  # 
                    #         if self.index_offset[i][j] % 2 != 0:
                    #             self.index_offset[i][j] += np.sign(self.index_offset[i][j])

                    # print(self.index_offset)

                    ### ODD2EVEN ###


                    ### EVEN2ODD ###

                    # perform theoretical shift in case of an odd number of shifts (to help repair alternating structures):
                    # this shifting should also include its own shift errors with the same probability
                    # -> we leave it out for now, just for theoretical testing
                    # -> in future, we could create a best-case and worst-case, in which latter would be that shift error happens also during this "correction"
                    # this would add some overhead in practice

                    # for i in range(0, self.index_offset.shape[0]):      # 
                    #     for j in range(0, self.index_offset.shape[1]):  # 
                    #         if self.index_offset[i][j] != 0 and self.index_offset[i][j] % 2 == 0:
                    #             self.index_offset[i][j] -= np.sign(self.index_offset[i][j])

                    # print(self.index_offset)

                    ### EVEN2ODD ###


                    ### AT RUNTIME ###

                    # # print(quantized_weight)
                    # endlen.apply_1flip(array_type="3D", block_size=self.block_size, data=quantized_weight)
                    # print("endlen flip applied")
                    # # print(quantized_weight)
                    
                    # # print(quantized_weight)
                    # if self.nr_run == 1:
                    #     endlen.apply_1flip_ind_off(array_type="3D", block_size=self.block_size, data=quantized_weight, index_offset=self.index_offset, global_bitflip_budget=self.global_bitflip_budget, local_bitflip_budget=self.local_bitflip_budget)
                    #     print("endlen flip according to index_offset applied")
                    #     self.q_weight = quantized_weight
                    # else:
                    #     quantized_weight = self.q_weight
                    # # print(quantized_weight)

                    ### AT RUNTIME ###

                    self.nr_run += 1


                quantized_weight = apply_error_model(quantized_weight, self.index_offset, self.block_size, self.error_model)
                

                if self.protectLayers[self.layerNR-1]==0:
                    differences = np.count_nonzero(quantized_weight_init.cpu() != quantized_weight.cpu())
                    self.bitflips[self.layerNR-1].append(differences)


                # if self.protectLayers[self.layerNR-1]==0:
                #     list_of_integers = quantized_weight.cpu().tolist()

                #     # Open a file in write mode
                #     with open('qweights_shift_'+str(self.layerNR)+'.txt', 'w') as f:
                #         f.write("[")
                #         # Write the list of integers to the file
                #         for integer in list_of_integers[:-1]:
                #             f.write(str(integer) + ',\n')
                #         f.write(str(list_of_integers[-1]) + "]")

                ###

                # if self.protectLayers[self.layerNR-1]==0:
                #     list_of_integers = quantized_weight.cpu().tolist()
                #     try:
                #         with open('qweights/'+str(self.block_size)+'/qweights_'+str(self.error_model.p)+'/qweights_shift1_'+str(self.layerNR)+'.txt', 'x') as f:
                #             f.write("[")
                #             # Write the list of integers to the file
                #             for integer in list_of_integers[:-1]:
                #                 f.write(str(integer) + ',\n')
                #             f.write(str(list_of_integers[-1]) + "]")
                #             print("Wrote content to shift1 file")
                #     except FileExistsError:
                #         print("shift1 already exists, writing to shift2 file.")
                #         try:
                #             with open('qweights/'+str(self.block_size)+'/qweights_'+str(self.error_model.p)+'/qweights_shift2_'+str(self.layerNR)+'.txt', 'x') as f:
                #                 f.write("[")
                #                 # Write the list of integers to the file
                #                 for integer in list_of_integers[:-1]:
                #                     f.write(str(integer) + ',\n')
                #                 f.write(str(list_of_integers[-1]) + "]")
                #                 print("Wrote content to shift2 file")
                #         except FileExistsError:
                #             print("shift2 already exists, skipping write.")
                        
                #     try:
                #         with open('qweights/'+str(self.block_size)+'/qweights_'+str(self.error_model.p)+'/qweights_shift10_'+str(self.layerNR)+'.txt', 'w') as f:
                #             f.write("[")
                #             # Write the list of integers to the file
                #             for integer in list_of_integers[:-1]:
                #                 f.write(str(integer) + ',\n')
                #             f.write(str(list_of_integers[-1]) + "]")
                #             print("Wrote content to shift10 file")
                #     except FileExistsError:
                #         print("shift10 already exists, skipping write.")

                

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
