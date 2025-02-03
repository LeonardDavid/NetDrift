import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import numpy as np
import random

import os
import time
import signal
import math

import metrics.ratio_blocks_ecc_ind_off.ratio_blocks_ecc_ind_off as ratio_blocks_io
import metrics.binomial_revert.binomial_revert as bin_revert
import metrics.blockhyp_endlen.blockhyp_endlen as blockhyp

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
    def forward(ctx, input, index_offset, rt_size=64, error_model=None):
        output = input.clone().detach()
        
        if error_model.__class__.__name__ == 'RacetrackModel':
            output = error_model.applyErrorModel(output, index_offset, rt_size)
        elif error_model.__class__.__name__ == 'BinarizeFIModel':
            output = error_model.applyErrorModel(output)
        else:
            print(f"Invalid error model {error_model}")
            exit()        

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


# add for compatibility to every apply_error_model parameters that do not use index_offset and rt_size
index_offset_default = np.zeros([2,2])
rt_size_default = 1.0


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
            output = apply_error_model(output, index_offset_default, rt_size_default, self.error_model)
        return output


# read flags.conf
def read_flags(file_path):
    flags = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Ignore comments and blank lines
            if line.startswith("#") or not line.strip():
                continue
            
            # Parse key-value pairs
            key, value = line.strip().split("=", 1)
            flags[key] = value
    return flags

script_dir = os.path.dirname(os.path.abspath(__file__))
flags_file = os.path.join(script_dir, "../..", "flags.conf")

flags = read_flags(flags_file)

# Check only Execution/Read flags
exec_keys = {key: value for key, value in flags.items() if key.startswith("EXEC_")}
read_keys = {key: value for key, value in flags.items() if key.startswith("READ_")}
true_count_exec = sum(1 for value in exec_keys.values() if value == "True")
true_count_read = sum(1 for value in read_keys.values() if value == "True")

# Assert whether there is at most one Execution flag turned on at the same time
assert true_count_exec <= 1, f"\n\033[0;31mMore than one Execution flag in flags.conf has the value 'True': {exec_keys}\n\033[0m"
print("Assertion passed: At most one Execution flag in flags.conf has the value 'True'.\n")

# Assert whether there is at most one Execution flag turned on at the same time
assert true_count_read <= 1, f"\n\033[0;31mMore than one Read flag in flags.conf has the value 'True': {read_keys}\n\033[0m"
print("Assertion passed: At most one Read flag in flags.conf has the value 'True'.\n")


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
        # self.kernel_size = kwargs.pop('kernel_size', None)
        self.index_offset = kwargs.pop('index_offset', None)
        self.rt_size = kwargs.pop('rt_size', None)
        self.protectLayers = kwargs.pop('protectLayers', None)
        self.affected_rts = kwargs.pop('affected_rts', None)
        self.misalign_faults = kwargs.pop('misalign_faults', None)
        self.bitflips = kwargs.pop('bitflips', None)
        self.global_bitflip_budget = kwargs.pop('global_bitflip_budget', None)
        self.local_bitflip_budget = kwargs.pop('local_bitflip_budget', None)

        self.calc_results = kwargs.pop('calc_results', None)
        self.calc_bitflips = kwargs.pop('calc_bitflips', None)
        self.calc_misalign_faults = kwargs.pop('calc_misalign_faults', None)
        self.calc_affected_rts = kwargs.pop('calc_affected_rts', None)
        
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


            ### PRNT ###

            if flags.get("PRNT_QWEIGHTS_BEFORE") == "True" and self.test_rtm is not None and self.protectLayers[self.layerNR-1]==0:
                list_of_integers = quantized_weight.cpu().tolist()

                try:
                    with open('qweights_orig_'+str(self.layerNR)+'.txt', 'w') as f:
                        f.write("[")
                        # Write the list of integers to the file
                        for integer in list_of_integers[:-1]:
                            f.write(str(integer) + ',\n')
                        f.write(str(list_of_integers[-1]) + "]")

                except FileExistsError:
                    print("Orig qweights file already exists")


            ### RTM_SIM ###

            if self.error_model is not None:
                if self.test_rtm is not None and self.protectLayers[self.layerNR-1]==0:
                    if flags.get("PRNT_LAYER_NAME") == "True":
                        print("Linear", self.layerNR)
                    # print(self.nr_run)

                    ### Read input qweight tensor from file ###

                    file = "Info: no input qweight tensor has been read"
                    data_tensor = np.array([])

                    # file = "metrics/count_len/q_in/qweights_orig_"+str(self.layerNR)+".txt"
                    # file = "metrics/count_len/q_out_indiv/qweights_orig_"+str(self.layerNR)+"_flip_"+str(n_l_r)+"_"+str(n_l_r)+".txt"
                    # file = "metrics/count_len/q_out/qweights_orig_"+str(self.layerNR)+"_"+str(nr_flip)+"flip"+str(bitlen)+"_"+str(n_l_r)+"_"+str(n_l_r)+".txt"
                    
                    ### else cases are handled by assertion outside of the class, after reading flags out

                    if flags.get("READ_ECC") == "True":
                        file = "metrics/ratio_blocks_ecc/" + flags.get("FOLDER_ECC") + "/qweights_ratio_ecc"+str(self.layerNR)+".txt"
                        
                        data_tensor = read_data(file).cuda()
                        quantized_weight = quantize(data_tensor, self.quantization)

                    if flags.get("READ_ECC_IND_OFF") == "True":
                        file = "metrics/ratio_blocks_ecc_ind_off/" + flags.get("FOLDER_ECC_IND_OFF") + "/qweights_ratio_ecc"+str(self.layerNR)+".txt"

                        data_tensor = read_data(file).cuda()
                        quantized_weight = quantize(data_tensor, self.quantization)

                    if flags.get("READ_ENDLEN") == "True":
                        file = "metrics/count_len/" + flags.get("FOLDER_ENDLEN") + "/qweights_orig_"+str(self.layerNR)+"_1flip_" + flags.get("TYPE_ENDLEN") + ".txt"

                        data_tensor = read_data(file).cuda()
                        quantized_weight = quantize(data_tensor, self.quantization)

                    if flags.get("PRNT_INPUT_FILE_INFO") == "True":
                        print(file)
                        print(data_tensor.shape) 
                        # L3: [2048, 3136]
                        # L4: [10, 2048]

                    ### calculate index offset for misalignment faults

                    quantized_weight_init = quantized_weight

                    misalign_fault = 0   # number of misalignment faults
                    shift = 0       # number of shifts (used for reading)
                    for i in range(0, self.index_offset.shape[0]):
                        for j in range(0, self.index_offset.shape[1]):
                            # start at 1 because AP is on the first element at the beginning, no shift is needed for reading the first value
                            for k in range(1, self.rt_size):
                                shift += 1
                                if(random.uniform(0.0, 1.0) < self.error_model.p):
                                    misalign_fault += 1
                                    # 50/50 random possibility (uniform distribution) of right or left misalign_fault
                                    if(random.choice([-1,1]) == 1):
                                        # right misalign_fault
                                        if (self.index_offset[i][j] < self.rt_size/2): # +1
                                            self.index_offset[i][j] += 1
                                        # self.index_offset[i][j] += 1
                                        # if (self.index_offset[i][j] > self.rt_size/2): # +1
                                        #     self.lost_vals_r[i][j] += 1
                                        #     quantized_weight[i][(j+1)*self.rt_size - int(self.lost_vals_r[i][j])] = random.choice([-1,1])
                                        # if (self.lost_vals_l[i][j] > 0):
                                        #     self.lost_vals_l[i][j] -= 1
                                    else:
                                        # left misalign_fault
                                        if (self.index_offset[i][j] < self.rt_size/2): # -1
                                            self.index_offset[i][j] -= 1
                                        # self.index_offset[i][j] -= 1
                                        # if(-self.index_offset[i][j] > self.rt_size/2): # +1
                                        #     self.lost_vals_l[i][j] += 1
                                        #     quantized_weight[i][j*self.rt_size + int(self.lost_vals_l[i][j]) - 1] = random.choice([-1,1])
                                        # if(self.lost_vals_r[i][j] > 0):
                                        #     self.lost_vals_r[i][j] -= 1

                    # self.misalign_faults_abs[self.layerNR-1] += misalign_fault
                    # print("local misalign_faults_abs: " + str(misalign_fault) + "/" + str(shift))
                    # print(self.misalign_faults_abs)

                    if self.calc_misalign_faults == "True":
                        self.misalign_faults[self.layerNR-1].append(misalign_fault)
                        # print(self.misalign_faults)


                    ### PRNT ###

                    if flags.get("PRNT_IND_OFF_BEFORE") == "True" and self.nr_run==1:
                        with open("ind_off_"+str(self.layerNR)+"_run_0.txt", "w") as f:
                            for i in range(0, self.index_offset.shape[0]):      # 
                                for j in range(0, self.index_offset.shape[1]):  #
                                    f.write(str(self.index_offset[i][j]) + " ")
                                f.write("\n")


                    ### BINOMIAL REVERT ###

                    # reset some index offset values above a certain threshold
                    # quite theoretical as well, because this would mean that error correction is applied only to some blocks, but in practice it is either full ecc or no ecc
                    # some possible thresholds: cut 80% of the amount of values starting from the middle (0, 1, -1, 2, -2 etc) and leave 20% on the edges
                    # or possibility 2: cut 80% of the total sizes starting from the edges (40% on the right, 40% on the left)
                    # significant overhead to be reckoned with, only for counting (and creating histogram)

                    if flags.get("EXEC_BIN_REVERT_MID") == "True":
                        # 80/20 from middle (total elements)
                        # if self.nr_run == 1:
                        self.index_offset = bin_revert.revert_elements_2d_mid(self.index_offset)
                    
                    if flags.get("EXEC_BIN_REVERT_EDGES") == "True":
                        # 80/20 from edges (total bins)
                        # if self.nr_run == 1:
                        self.index_offset = bin_revert.revert_elements_2d_edges(self.index_offset)


                    ### ODD2EVEN ###

                    # perform theoretical shift in case of an odd number of shifts (to help repair alternating structures):
                    # this shifting should also include its own shift errors with the same probability
                    # -> we leave it out for now, just for theoretical testing
                    # -> in future, we could create a best-case and worst-case, in which latter would be that shift error happens also during this "correction"
                    # this would add some overhead in practice

                    if flags.get("EXEC_ODD2EVEN_DEC") == "True":
                        for i in range(0, self.index_offset.shape[0]):      
                            for j in range(0, self.index_offset.shape[1]):  
                                if self.index_offset[i][j] % 2 != 0:
                                    self.index_offset[i][j] -= np.sign(self.index_offset[i][j])

                    if flags.get("EXEC_ODD2EVEN_INC") == "True":
                        for i in range(0, self.index_offset.shape[0]):      
                            for j in range(0, self.index_offset.shape[1]):  
                                if self.index_offset[i][j] % 2 != 0:
                                    self.index_offset[i][j] += np.sign(self.index_offset[i][j])


                    ### EVEN2ODD ###

                    # perform theoretical shift in case of an odd number of shifts (to help repair alternating structures):
                    # this shifting should also include its own shift errors with the same probability
                    # -> we leave it out for now, just for theoretical testing
                    # -> in future, we could create a best-case and worst-case, in which latter would be that shift error happens also during this "correction"
                    # this would add some overhead in practice

                    if flags.get("EXEC_EVEN2ODD_DEC") == "True":
                        for i in range(0, self.index_offset.shape[0]):      
                            for j in range(0, self.index_offset.shape[1]):  
                                if self.index_offset[i][j] != 0 and self.index_offset[i][j] % 2 == 0:
                                    self.index_offset[i][j] -= np.sign(self.index_offset[i][j])

                    if flags.get("EXEC_EVEN2ODD_INC") == "True":
                        for i in range(0, self.index_offset.shape[0]):      
                            for j in range(0, self.index_offset.shape[1]):  
                                if self.index_offset[i][j] != 0 and self.index_offset[i][j] % 2 == 0:
                                    self.index_offset[i][j] += np.sign(self.index_offset[i][j])
                    

                    ### AT RUNTIME ###

                    ### #RATIO_BLOCKS_IND_OFF# ###
                    if flags.get("EXEC_RATIO_BLOCKS_IND_OFF") == "True":
                        if self.nr_run == 1:
                            ratio_blocks_io.apply_ratio_ind_off(array_type="1D", rt_size=self.rt_size, data=quantized_weight, index_offset=self.index_offset, global_bitflip_budget=self.global_bitflip_budget, local_bitflip_budget=self.local_bitflip_budget)
                            print("ratio_blocks flip according to index_offset applied")
                            self.q_weight = quantized_weight
                        else:
                            quantized_weight = self.q_weight


                    ### #ENDLEN# ###
                    if flags.get("EXEC_ENDLEN") == "True":

                        qweight_initial_shape = quantized_weight.shape
                        # print(f"qweight initial shape: {qweight_initial_shape}")

                        quantized_weight = quantized_weight.clone().view(-1)
                        # print(f"qweight reshaped: {quantized_weight.shape}")

                        # start_time = time.time()
                        # blockhyp.blockhyp_endlen_algorithm(data=quantized_weight, rt_size=self.rt_size)
                        blockhyp.blockhyp_endlen_algorithm_parallel(data=quantized_weight, rt_size=self.rt_size)
                        # end_time = time.time()
                        # print(f"Time taken for blockhyp.blockhyp_endlen_algorithm: {end_time - start_time} seconds")

                        quantized_weight = quantized_weight.view(qweight_initial_shape)
                        # print(f"qweight reshaped back: {quantized_weight.shape}")

                        # # Interrupt the code execution immediately
                        # os.kill(os.getpid(), signal.SIGINT)
                               

                    ### #ENDLEN IND_OFF# ###
                    if flags.get("EXEC_ENDLEN_IND_OFF") == "True":
                        if self.nr_run == 1:
                            endlen.apply_1flip_ind_off(array_type="1D", rt_size=self.rt_size, data=quantized_weight, index_offset=self.index_offset, global_bitflip_budget=self.global_bitflip_budget, local_bitflip_budget=self.local_bitflip_budget)
                            print("endlen flip according to index_offset applied")
                            self.q_weight = quantized_weight
                        else:
                            quantized_weight = self.q_weight

                    ### AT RUNTIME ###

                    self.nr_run += 1

                    ### PRNT ###

                    if flags.get("PRNT_IND_OFF_AFTER") == "True" and flags.get("PRNT_IND_OFF_AFTER_NRUN") == str(self.nr_run-1):
                        with open("ind_off_"+str(self.layerNR)+"_run_"+str(self.nr_run-1)+".txt", "w") as f:
                            for i in range(0, self.index_offset.shape[0]):      # 
                                for j in range(0, self.index_offset.shape[1]):  #
                                    f.write(str(self.index_offset[i][j]) + " ")
                                f.write("\n")


                quantized_weight = ErrorModel.apply(quantized_weight, self.index_offset, self.rt_size, self.error_model)


                if self.test_rtm is not None and self.protectLayers[self.layerNR-1]==0:

                    ## absolute number of bitflips
                    if self.calc_bitflips == "True":
                        differences = np.count_nonzero(quantized_weight_init.cpu() != quantized_weight.cpu())
                        self.bitflips[self.layerNR-1].append(differences)

                    ## number of misaligned racetracks
                    if self.calc_affected_rts == "True":
                        affected_racetracks = np.count_nonzero(self.index_offset)
                        self.affected_rts[self.layerNR-1].append(affected_racetracks)


                ### PRNT ###

                if flags.get("PRNT_QWEIGHTS_AFTER") == "True" and self.test_rtm is not None and self.protectLayers[self.layerNR-1]==0 and flags.get("PRNT_QWEIGHTS_AFTER_NRUN") == str(self.nr_run-1):
                    list_of_integers = quantized_weight.cpu().tolist()

                    with open('qweights_shift' + str(self.nr_run-1) + '_' + str(self.layerNR) + '.txt', 'w') as f:
                        f.write("[")
                        # Write the list of integers to the file
                        for integer in list_of_integers[:-1]:
                            f.write(str(integer) + ',\n')
                        f.write(str(list_of_integers[-1]) + "]")


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
                quantized_weight = apply_error_model(quantized_weight, index_offset_default, rt_size_default, self.error_model)
                quantized_bias = apply_error_model(quantized_bias, index_offset_default, rt_size_default, self.error_model)
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
        # self.kernel_size = kwargs.pop('kernel_size', None)
        self.index_offset = kwargs.pop('index_offset', None)
        self.rt_size = kwargs.pop('rt_size', None)
        self.protectLayers = kwargs.pop('protectLayers', None)
        self.affected_rts = kwargs.pop('affected_rts', None)
        self.misalign_faults = kwargs.pop('misalign_faults', None)
        self.bitflips = kwargs.pop('bitflips', None)
        self.global_bitflip_budget = kwargs.pop('global_bitflip_budget', None)
        self.local_bitflip_budget = kwargs.pop('local_bitflip_budget', None)

        self.calc_results = kwargs.pop('calc_results', None)
        self.calc_bitflips = kwargs.pop('calc_bitflips', None)
        self.calc_misalign_faults = kwargs.pop('calc_misalign_faults', None)
        self.calc_affected_rts = kwargs.pop('calc_affected_rts', None)

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


            ### PRNT ###

            if flags.get("PRNT_QWEIGHTS_BEFORE") == "True" and self.test_rtm is not None and self.protectLayers[self.layerNR-1]==0:
                list_of_integers = quantized_weight.cpu().tolist()

                try:
                    with open('qweights_orig_'+str(self.layerNR)+'.txt', 'w') as f:
                        f.write("[")
                        # Write the list of integers to the file
                        for integer in list_of_integers[:-1]:
                            f.write(str(integer) + ',\n')
                        f.write(str(list_of_integers[-1]) + "]")

                except FileExistsError:
                    print("Orig qweights file already exists")


            ### RTM_SIM ###

            if self.error_model is not None:

                if self.test_rtm is not None and self.protectLayers[self.layerNR-1]==0:
                    if flags.get("PRNT_LAYER_NAME") == "True":
                        print("Convolution2D", self.layerNR)
                    # print(self.nr_run)

                    ### Read input qweight tensor from file ###

                    file = "Info: no input qweight tensor has been read"
                    data_tensor = np.array([])

                    # file = "metrics/count_len/q_in/qweights_orig_"+str(self.layerNR)+".txt"
                    # file = "metrics/count_len/q_out_indiv/qweights_orig_"+str(self.layerNR)+"_flip_"+str(n_l_r)+"_"+str(n_l_r)+".txt"
                    # file = "metrics/count_len/q_out/qweights_orig_"+str(self.layerNR)+"_"+str(nr_flip)+"flip"+str(bitlen)+"_"+str(n_l_r)+"_"+str(n_l_r)+".txt"
                    
                    ### else cases are handled by assertion outside of the class, after reading flags out

                    if flags.get("READ_ECC") == "True":
                        file = "metrics/ratio_blocks_ecc/" + flags.get("FOLDER_ECC") + "/qweights_ratio_ecc"+str(self.layerNR)+".txt"
                        
                        data_tensor = read_data(file).cuda()
                        quantized_weight = quantize(data_tensor, self.quantization)

                    if flags.get("READ_ECC_IND_OFF") == "True":
                        file = "metrics/ratio_blocks_ecc_ind_off/" + flags.get("FOLDER_ECC_IND_OFF") + "/qweights_ratio_ecc"+str(self.layerNR)+".txt"

                        data_tensor = read_data(file).cuda()
                        quantized_weight = quantize(data_tensor, self.quantization)

                    if flags.get("READ_ENDLEN") == "True":
                        file = "metrics/count_len/" + flags.get("FOLDER_ENDLEN") + "/qweights_orig_"+str(self.layerNR)+"_1flip_" + flags.get("TYPE_ENDLEN") + ".txt"

                        data_tensor = read_data(file).cuda()
                        quantized_weight = quantize(data_tensor, self.quantization)

                    if flags.get("PRNT_INPUT_FILE_INFO") == "True":
                        print(file)
                        print(data_tensor.shape) 
                        # L1: [64, 1, 3, 3]
                        # L2: [64, 64, 3, 3]

                    ### calculate index offset for misalignment faults
                    
                    quantized_weight_init = quantized_weight

                    misalign_fault = 0   # number of misalignment faults
                    shift = 0       # number of shifts (used for reading)
                    # iterate over all blocks (row-wise -> swap for loops for column-wise)
                    for i in range(0, self.index_offset.shape[0]):
                        for j in range(0, self.index_offset.shape[1]):
                            # now, read every value from the block
                            # start at 1 because AP is on the first element at the beginning, no shift is needed for reading the first value
                            for k in range(1, self.rt_size):
                                shift += 1
                                if(random.uniform(0.0, 1.0) < self.error_model.p):
                                    # print(self.error_model.p)
                                    misalign_fault += 1
                                    # 50/50 random possibility of right or left misalign_fault
                                    if(random.choice([-1,1]) == 1):
                                        # right misalign_fault
                                        if (self.index_offset[i][j] < self.rt_size/2): # +1
                                            self.index_offset[i][j] += 1
                                        # self.index_offset[i][j] += 1
                                        # if (self.index_offset[i][j] > self.rt_size/2): # +1
                                        #     self.lost_vals_r[i][j] += 1
                                        #     quantized_weight[i][(j+1)*self.rt_size - int(self.lost_vals_r[i][j])] = random.choice([-1,1])
                                        # if (self.lost_vals_l[i][j] > 0):
                                        #     self.lost_vals_l[i][j] -= 1
                                    else:
                                        # left misalign_fault
                                        if (self.index_offset[i][j] < self.rt_size/2): # -1
                                            self.index_offset[i][j] -= 1
                                        # self.index_offset[i][j] -= 1
                                        # if(-self.index_offset[i][j] > self.rt_size/2): # +1
                                        #     self.lost_vals_l[i][j] += 1
                                        #     quantized_weight[i][j*self.rt_size + int(self.lost_vals_l[i][j]) - 1] = random.choice([-1,1])
                                        # if(self.lost_vals_r[i][j] > 0):
                                        #     self.lost_vals_r[i][j] -= 1

                    # self.misalign_faults_abs[self.layerNR-1] += misalign_fault
                    # print("local misalign_faults_abs: " + str(misalign_fault) + "/" + str(shift))
                    # print(self.misalign_faults_abs)

                    if self.calc_misalign_faults == "True":
                        self.misalign_faults[self.layerNR-1].append(misalign_fault)
                        # print(self.misalign_faults)


                    ### PRNT ###

                    if flags.get("PRNT_IND_OFF_BEFORE") == "True" and self.nr_run==1:
                        with open("ind_off_"+str(self.layerNR)+"_run_0.txt", "w") as f:
                            for i in range(0, self.index_offset.shape[0]):      # 
                                for j in range(0, self.index_offset.shape[1]):  #
                                    f.write(str(self.index_offset[i][j]) + " ")
                                f.write("\n")


                    ### BINOMIAL REVERT ###

                    # reset some index offset values above a certain threshold
                    # quite theoretical as well, because this would mean that error correction is applied only to some blocks, but in practice it is either full ecc or no ecc
                    # some possible thresholds: cut 80% of the amount of values starting from the middle (0, 1, -1, 2, -2 etc) and leave 20% on the edges
                    # or possibility 2: cut 80% of the total sizes starting from the edges (40% on the right, 40% on the left)
                    # significant overhead to be reckoned with, only for counting (and creating histogram)

                    if flags.get("EXEC_BIN_REVERT_MID") == "True":
                        # 80/20 from middle (total elements)
                        # if self.nr_run == 1:
                        self.index_offset = bin_revert.revert_elements_2d_mid(self.index_offset)
                    
                    if flags.get("EXEC_BIN_REVERT_EDGES") == "True":
                        # 80/20 from edges (total bins)
                        # if self.nr_run == 1:
                        self.index_offset = bin_revert.revert_elements_2d_edges(self.index_offset)


                    ### ODD2EVEN ###

                    # perform theoretical shift in case of an odd number of shifts (to help repair alternating structures):
                    # this shifting should also include its own shift errors with the same probability
                    # -> we leave it out for now, just for theoretical testing
                    # -> in future, we could create a best-case and worst-case, in which latter would be that shift error happens also during this "correction"
                    # this would add some overhead in practice
                    if flags.get("EXEC_ODD2EVEN_DEC") == "True":
                        for i in range(0, self.index_offset.shape[0]):      
                            for j in range(0, self.index_offset.shape[1]):  
                                if self.index_offset[i][j] % 2 != 0:
                                    self.index_offset[i][j] -= np.sign(self.index_offset[i][j])

                    if flags.get("EXEC_ODD2EVEN_INC") == "True":
                        for i in range(0, self.index_offset.shape[0]):      
                            for j in range(0, self.index_offset.shape[1]):  
                                if self.index_offset[i][j] % 2 != 0:
                                    self.index_offset[i][j] += np.sign(self.index_offset[i][j])


                    ### EVEN2ODD ###

                    # perform theoretical shift in case of an odd number of shifts (to help repair alternating structures):
                    # this shifting should also include its own shift errors with the same probability
                    # -> we leave it out for now, just for theoretical testing
                    # -> in future, we could create a best-case and worst-case, in which latter would be that shift error happens also during this "correction"
                    # this would add some overhead in practice
                    
                    if flags.get("EXEC_EVEN2ODD_DEC") == "True":
                        for i in range(0, self.index_offset.shape[0]):      
                            for j in range(0, self.index_offset.shape[1]):  
                                if self.index_offset[i][j] != 0 and self.index_offset[i][j] % 2 == 0:
                                    self.index_offset[i][j] -= np.sign(self.index_offset[i][j])

                    if flags.get("EXEC_EVEN2ODD_INC") == "True":
                        for i in range(0, self.index_offset.shape[0]):      
                            for j in range(0, self.index_offset.shape[1]):  
                                if self.index_offset[i][j] != 0 and self.index_offset[i][j] % 2 == 0:
                                    self.index_offset[i][j] += np.sign(self.index_offset[i][j])


                    ### AT RUNTIME ###

                    ### #RATIO_BLOCKS_IND_OFF# ###
                    if flags.get("EXEC_RATIO_BLOCKS_IND_OFF") == "True":
                        if self.nr_run == 1:
                            ratio_blocks_io.apply_ratio_ind_off(array_type="3D", rt_size=self.rt_size, data=quantized_weight, index_offset=self.index_offset, global_bitflip_budget=self.global_bitflip_budget, local_bitflip_budget=self.local_bitflip_budget)
                            print("ratio_blocks flip according to index_offset applied")
                            self.q_weight = quantized_weight
                        else:
                            quantized_weight = self.q_weight


                    ### #ENDLEN# ###
                    if flags.get("EXEC_ENDLEN") == "True":

                        qweight_initial_shape = quantized_weight.shape
                        # print(f"qweight initial shape: {qweight_initial_shape}")

                        quantized_weight = quantized_weight.clone().view(-1)
                        # print(f"qweight reshaped: {quantized_weight.shape}")

                        # start_time = time.time()
                        # blockhyp.blockhyp_endlen_algorithm(data=quantized_weight, rt_size=self.rt_size)
                        blockhyp.blockhyp_endlen_algorithm_parallel(data=quantized_weight, rt_size=self.rt_size)
                        # end_time = time.time()
                        # print(f"Time taken for blockhyp.blockhyp_endlen_algorithm: {end_time - start_time} seconds")

                        quantized_weight = quantized_weight.view(qweight_initial_shape)
                        # print(f"qweight reshaped back: {quantized_weight.shape}")

                        # # Interrupt the code execution immediately
                        # os.kill(os.getpid(), signal.SIGINT)

                    
                    ### #ENDLEN IND_OFF# ###
                    if flags.get("EXEC_ENDLEN_IND_OFF") == "True":
                        if self.nr_run == 1:
                            endlen.apply_1flip_ind_off(array_type="3D", rt_size=self.rt_size, data=quantized_weight, index_offset=self.index_offset, global_bitflip_budget=self.global_bitflip_budget, local_bitflip_budget=self.local_bitflip_budget)
                            print("endlen flip according to index_offset applied")
                            self.q_weight = quantized_weight
                        else:
                            quantized_weight = self.q_weight

                    ### AT RUNTIME ###

                    self.nr_run += 1

                    ### PRNT ###

                    if flags.get("PRNT_IND_OFF_AFTER") == "True" and flags.get("PRNT_IND_OFF_AFTER_NRUN") == str(self.nr_run-1):
                        with open("ind_off_"+str(self.layerNR)+"_run_"+str(self.nr_run-1)+".txt", "w") as f:
                            for i in range(0, self.index_offset.shape[0]):      # 
                                for j in range(0, self.index_offset.shape[1]):  #
                                    f.write(str(self.index_offset[i][j]) + " ")
                                f.write("\n")


                quantized_weight = apply_error_model(quantized_weight, self.index_offset, self.rt_size, self.error_model)
                

                if self.test_rtm is not None and self.protectLayers[self.layerNR-1]==0:

                    ## absolute number of bitflips             
                    if self.calc_bitflips == "True":
                        differences = np.count_nonzero(quantized_weight_init.cpu() != quantized_weight.cpu())
                        self.bitflips[self.layerNR-1].append(differences)

                    ## number of misaligned racetracks
                    if self.calc_affected_rts == "True":
                        affected_racetracks = np.count_nonzero(self.index_offset)
                        self.affected_rts[self.layerNR-1].append(affected_racetracks)


                ### PRNT ###

                if flags.get("PRNT_QWEIGHTS_AFTER") == "True" and self.test_rtm is not None and self.protectLayers[self.layerNR-1]==0 and flags.get("PRNT_QWEIGHTS_AFTER_NRUN") == str(self.nr_run-1):
                    list_of_integers = quantized_weight.cpu().tolist()

                    with open('qweights_shift' + str(self.nr_run-1) + '_' + str(self.layerNR) + '.txt', 'w') as f:
                        f.write("[")
                        # Write the list of integers to the file
                        for integer in list_of_integers[:-1]:
                            f.write(str(integer) + ',\n')
                        f.write(str(list_of_integers[-1]) + "]")


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
                quantized_weight = apply_error_model(quantized_weight, index_offset_default, rt_size_default, self.error_model)
                quantized_bias = apply_error_model(quantized_bias, index_offset_default, rt_size_default, self.error_model)
            # compute regular 2d conv
            output = F.conv2d(input, quantized_weight, quantized_bias, self.stride,
                              self.padding, self.dilation, self.groups)
            return output
