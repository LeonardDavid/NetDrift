import os
import signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import numpy as np

from cuda.racetrack import racetrack_sim

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
    def forward(ctx, input, error_model=None):
        output = input.clone().detach()
        
        if error_model.__class__.__name__ == 'BinarizeFIModel':
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
        filepath: 
            Path to the file containing the data.
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


# Read and process flags.conf
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

# # Assert whether there is at most one Execution flag turned on at the same time
# assert true_count_exec <= 1, f"\n\033[0;31mMore than one Execution flag in flags.conf has the value 'True': {exec_keys}\n\033[0m"
# print("Assertion passed: At most one Execution flag in flags.conf has the value 'True'.\n")

# Assert whether there is at most one Read flag turned on at the same time
assert true_count_read <= 1, f"\n\033[0;31mMore than one Read flag in flags.conf has the value 'True': {read_keys}\n\033[0m"
print("Assertion passed: At most one Read flag in flags.conf has the value 'True'.\n")


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
            output = apply_error_model(output, self.error_model)
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

        # self.kernel_size = kwargs.pop('kernel_size', None)
        self.test_rtm = kwargs.pop('test_rtm', False)
        self.rt_error = kwargs.pop('rt_error', None)
        self.rt_mapping = kwargs.pop('rt_mapping', None)
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
        super(QuantizedLinear, self).__init__(*args, **kwargs)

    def forward(self, input):

        if self.bias is None:
            quantized_weight = None
            check_q = check_quantization(self.quantize_train, self.quantize_eval, self.training)

            if (check_q == True):
                quantized_weight = quantize(self.weight, self.quantization)
            else:
                quantized_weight = self.weight


            # If the error model is not None, apply a standard error model to the quantized_weight tensor
            if self.error_model is not None:
                quantized_weight = apply_error_model(quantized_weight, self.error_model)
            
            # If the error model has not been specified, probably RTM simulation is desired
            else:
                # If the test_rtm flag is set to 1, perform the RTM simulation for unprotected layers
                if self.test_rtm == 1:
                    
                    # If the layer is unprotected, perform the RTM simulation
                    if self.protectLayers[self.layerNR-1] == 0:

                        ### PRNT ###

                        if flags.get("PRNT_LAYER_NAME") == "True":
                            print("Linear", self.layerNR)

                        if flags.get("PRNT_QWEIGHTS_BEFORE") == "True":
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

                        ### Read input qweight tensor from file ###

                        file = "Info: no input qweight tensor has been read"
                        data_tensor = np.array([])

                        ## else cases are handled by assertion outside of the class, after reading flags out
                        if flags.get("READ_ENDLEN") == "True":
                            file = "metrics/blockhyp_endlen/" + flags.get("FOLDER_ENDLEN") + "/qweights_orig_"+str(self.layerNR)+"_1flip.txt"
                            
                            data_tensor = read_data(file).cuda()
                            quantized_weight = quantize(data_tensor, self.quantization)
                        
                        if flags.get("READ_TEST") == "True":
                            file = "metrics/blockhyp_endlen/" + flags.get("FOLDER_TEST") + "/qweights_"+str(self.layerNR)+".txt"
                            
                            data_tensor = read_data(file).cuda()
                            quantized_weight = quantize(data_tensor, self.quantization)

                        if flags.get("PRNT_INPUT_FILE_INFO") == "True":
                            print(file)
                            print(data_tensor.shape) 
                            # L3: [2048, 3136]
                            # L4: [10, 2048]

                        ### Main RTM Simulation ###

                        ## Needed to calculate absolute number of bitflips
                        if self.calc_bitflips == "True":
                            quantized_weight_init = quantized_weight

                        ## Specify the number of shifts required for the Access Port (AP) to read a single word from the racetrack
                        if self.rt_mapping == "ROW":
                            # When data is mapped row-wise, the AP requires rt_size shifts to read a single word L_i, i.e. 1 bit per shift
                            ap_reads = self.rt_size * self.rt_size
                            shifts = ap_reads - self.rt_size
                        elif self.rt_mapping == "COL":
                            # When data is mapped column-wise, the AP reads bit b_i from each racetrack, i.e. an entire word L_i per shift (e.g. multiple convolutions)
                            ap_reads = self.rt_size
                            shifts = ap_reads - 1
                        else:
                            raise ValueError(f"Invalid racetrack mapping: {self.rt_mapping}")
                        
                        # Transpose quantized_weight if rt_mapping is set to column-wise, in order to have the same internal representation
                        # This is necessary in order for the racetrack simulation kernel to execute the same allocations and operations, regardless of rt_mapping
                        if self.rt_mapping == "COL":
                            # Make sure it is contiguous -> requirement for racetrack kernel
                            quantized_weight = quantized_weight.t().contiguous()

                        ## Compute index_offset and read out the misaligned quantized_weight based on index_offset for each racetracks
                        quantized_weight, self.index_offset, misalign_fault = racetrack_sim(quantized_weight=quantized_weight.clone(), index_offset=self.index_offset, 
                                                                                                                    rt_size=self.rt_size, rt_error=self.rt_error, ap_reads=ap_reads,
                                                                                                                    flags=flags, nr_run=self.nr_run, layerNR=self.layerNR)
                        # Transpose quantized_weight back to original shape if rt_mapping is set to column-wise 
                        # This is necessary for the following operations to work correctly (and in case rt_mapping is mixed and the following layer has different mapping)
                        if self.rt_mapping == "COL":
                            quantized_weight = quantized_weight.t()


                        ## Number of misalignment faults 
                        if self.calc_misalign_faults == "True":
                            self.misalign_faults[self.layerNR-1].append(misalign_fault)

                        ## Absolute number of bitflips
                        if self.calc_bitflips == "True":
                            differences = np.count_nonzero(quantized_weight_init.cpu() != quantized_weight.cpu())
                            self.bitflips[self.layerNR-1].append(differences)

                        ## Number of misaligned racetracks
                        if self.calc_affected_rts == "True":
                            affected_racetracks = np.count_nonzero(self.index_offset)
                            self.affected_rts[self.layerNR-1].append(affected_racetracks)

                        ### PRNT ###

                        if flags.get("PRNT_QWEIGHTS_AFTER") == "True" and flags.get("PRNT_QWEIGHTS_AFTER_NRUN") == str(self.nr_run):
                            list_of_integers = quantized_weight.cpu().tolist()

                            with open('qweights_shift' + str(self.nr_run) + '_' + str(self.layerNR) + '.txt', 'w') as f:
                                f.write("[")
                                # Write the list of integers to the file
                                for integer in list_of_integers[:-1]:
                                    f.write(str(integer) + ',\n')
                                f.write(str(list_of_integers[-1]) + "]")
                        
                        self.nr_run += 1
                    else:
                        # If the layer is protected, quantized_weight remains unchanged
                        pass
                else:
                    # If the RTM Simulation flag is not set, quantized_weight remains unchanged
                    pass

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
                quantized_weight = apply_error_model(quantized_weight, self.error_model)
                quantized_bias = apply_error_model(quantized_bias, self.error_model)
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

        # self.kernel_size = kwargs.pop('kernel_size', None)
        self.test_rtm = kwargs.pop('test_rtm', False)
        self.rt_error = kwargs.pop('rt_error', None)
        self.rt_mapping = kwargs.pop('rt_mapping', None)
        self.kernel_mapping = kwargs.pop('kernel_mapping', None)
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
        super(QuantizedConv2d, self).__init__(*args, **kwargs)

    def forward(self, input):

        if self.bias is None:
            quantized_weight = None
            check_q = check_quantization(self.quantize_train, self.quantize_eval, self.training)

            if (check_q == True):
                quantized_weight = quantize(self.weight, self.quantization)
            else:
                quantized_weight = self.weight
                quantized_bias = self.bias


            # If the error model is not None, apply a standard error model to the quantized_weight tensor
            if self.error_model is not None:
                quantized_weight = apply_error_model(quantized_weight, self.error_model)
            
            # If the error model has not been specified, probably RTM simulation is desired
            else:
                # If the test_rtm flag is set to 1, perform the RTM simulation for unprotected layers
                if self.test_rtm == 1:
                    
                    # If the layer is unprotected, perform the RTM simulation
                    if self.protectLayers[self.layerNR-1]==0:

                        ### PRNT ###

                        if flags.get("PRNT_LAYER_NAME") == "True":
                            print("Convolution2D", self.layerNR)

                        if flags.get("PRNT_QWEIGHTS_BEFORE") == "True":
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

                        ### Read input qweight tensor from file ###

                        file = "Info: no input qweight tensor has been read"
                        data_tensor = np.array([])

                        ## else cases are handled by assertion outside of the class, after reading flags out
                        if flags.get("READ_ENDLEN") == "True":
                            file = "metrics/blockhyp_endlen/" + flags.get("FOLDER_ENDLEN") + "/qweights_orig_"+str(self.layerNR)+"_1flip.txt"

                            data_tensor = read_data(file).cuda()
                            quantized_weight = quantize(data_tensor, self.quantization)

                        if flags.get("READ_TEST") == "True":
                            file = "metrics/blockhyp_endlen/" + flags.get("FOLDER_TEST") + "/qweights_"+str(self.layerNR)+".txt"
                            
                            data_tensor = read_data(file).cuda()
                            quantized_weight = quantize(data_tensor, self.quantization)

                        if flags.get("PRNT_INPUT_FILE_INFO") == "True":
                            print(file)
                            print(data_tensor.shape) 
                            # L1: [64, 1, 3, 3]
                            # L2: [64, 64, 3, 3]

                        ### Main RTM Simulation ###

                        ## Needed to calculate absolute number of bitflips
                        if self.calc_bitflips == "True":
                            quantized_weight_init = quantized_weight
                        
                        ## Specify the number of shifts required for the Access Port (AP) to read a single word from the racetrack
                        if self.rt_mapping == "ROW":
                            # When data is mapped row-wise, the AP requires rt_size shifts to read a single word L_i, i.e. 1 bit per shift
                            ap_reads = self.rt_size * self.rt_size
                            shifts = ap_reads - self.rt_size
                        elif self.rt_mapping == "COL":
                            # When data is mapped column-wise, the AP reads bit b_i from each racetrack, i.e. an entire word L_i per shift (e.g. multiple convolutions)
                            ap_reads = self.rt_size
                            shifts = ap_reads - 1
                        else:
                            raise ValueError(f"Invalid racetrack mapping: {self.rt_mapping}")

                        # TODO tests
                        # Original shape is [out_channels, in_channels, self.kernel_size, self.kernel_size]
                        quantized_weight_initial_shape = quantized_weight.shape

                        # If the kernel mapping is specified, rearrange the weights in each kernel in the specified order
                        match self.kernel_mapping:
                            case "ROW":
                                # default order/rev_order (not needed)
                                pass
                            case "COL":
                                order = col_indices_3x3
                                rev_order = reverse_col_indices_3x3 
                            case "CLW":
                                order = clw_indices_3x3
                                rev_order = reverse_clw_indices_3x3
                            case "ACW":
                                order = acw_indices_3x3
                                rev_order = reverse_acw_indices_3x3
                            case _:
                                raise ValueError(f"Invalid kernel mapping: {self.kernel_mapping}")
                            
                        # print("quantized_weight before:", quantized_weight)
                        # Rearrange the weights in each kernel in specified order (COLumnwise, CLockWise, AntiClockWise) -> ROWwise is default
                        if self.kernel_mapping != "ROW":
                            quantized_weight = rearrange_kernel_weights(quantized_weight, order)

                        # Reshape quantized_weight to 2D tensor, since convolutional weights are 4D
                        # This is necessary for the racetrack simulation kernel to execute the same allocations and operations, regardless of the input shape
                        quantized_weight = quantized_weight.view(quantized_weight.size(0), -1)
                        
                        # Transpose quantized_weight if rt_mapping is set to column-wise, in order to have the same internal representation
                        # This is necessary in order for the racetrack simulation kernel to execute the same allocations and operations, regardless of rt_mapping
                        if self.rt_mapping == "COL":
                            # Make sure it is contiguous -> requirement for racetrack kernel
                            quantized_weight = quantized_weight.t().contiguous()
                            
                        ## Compute index_offset and read out the misaligned quantized_weight based on index_offset for each racetracks
                        quantized_weight, self.index_offset, misalign_fault = racetrack_sim(quantized_weight=quantized_weight.clone(), index_offset=self.index_offset, 
                                                                                                                    rt_size=self.rt_size, rt_error=self.rt_error, ap_reads=ap_reads,
                                                                                                                    flags=flags, nr_run=self.nr_run, layerNR=self.layerNR)

                        # Transpose quantized_weight back to original shape if rt_mapping is set to column-wise 
                        # This is necessary for the following operations to work correctly (and in case rt_mapping is mixed and the following layer has different mapping)
                        if self.rt_mapping == "COL":
                            quantized_weight = quantized_weight.t()

                        # If each kernel has been rearranged previously in a specified order, rearrange them back to the original order
                        if self.kernel_mapping != "ROW":
                            # First reshape 2D back to 3D [out_channels, in_channels, self.kernel_size * self.kernel_size]
                            quantized_weight = quantized_weight.view(quantized_weight_initial_shape[0], 
                                                                quantized_weight_initial_shape[1], 
                                                                quantized_weight_initial_shape[2] * quantized_weight_initial_shape[3])
                            
                            # Then rearrange back to initial order and reshape tensor back to 4D
                            quantized_weight = quantized_weight[..., rev_order].view(quantized_weight_initial_shape)
                        else:
                            # If instead the default ROWwise mapping was used for the kernel, then simply reshape from 2D->4D to initial shape (without changes due to kernel mapping)
                            # This is necessary for the following operations to work correctly
                            quantized_weight = quantized_weight.view(quantized_weight_initial_shape)


                        ## Number of misalignment faults
                        if self.calc_misalign_faults == "True":
                            self.misalign_faults[self.layerNR-1].append(misalign_fault)
                        
                        ## Absolute number of bitflips
                        if self.calc_bitflips == "True":
                            differences = np.count_nonzero(quantized_weight_init.cpu() != quantized_weight.cpu())
                            self.bitflips[self.layerNR-1].append(differences)

                        ## Number of misaligned racetracks
                        if self.calc_affected_rts == "True":
                            affected_racetracks = np.count_nonzero(self.index_offset)
                            self.affected_rts[self.layerNR-1].append(affected_racetracks)

                        ### PRNT ###

                        if flags.get("PRNT_QWEIGHTS_AFTER") == "True" and flags.get("PRNT_QWEIGHTS_AFTER_NRUN") == str(self.nr_run):
                            list_of_integers = quantized_weight.cpu().tolist()

                            with open('qweights_shift' + str(self.nr_run) + '_' + str(self.layerNR) + '.txt', 'w') as f:
                                f.write("[")
                                # Write the list of integers to the file
                                for integer in list_of_integers[:-1]:
                                    f.write(str(integer) + ',\n')
                                f.write(str(list_of_integers[-1]) + "]")
                        
                        self.nr_run += 1
                    else:
                        # If the layer is protected, quantized_weight remains unchanged
                        pass
                else:
                    # If the RTM Simulation flag is not set, quantized_weight remains unchanged
                    pass


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
                quantized_weight = apply_error_model(quantized_weight, self.error_model)
                quantized_bias = apply_error_model(quantized_bias, self.error_model)
            # compute regular 2d conv
            output = F.conv2d(input, quantized_weight, quantized_bias, self.stride,
                              self.padding, self.dilation, self.groups)
            return output


# Define rearrangement indices for a 3x3 kernel
col_indices_3x3 = torch.tensor([0, 3, 6, 1, 4, 7, 2, 5, 8])
clw_indices_3x3 = torch.tensor([0, 1, 2, 5, 8, 7, 6, 3, 4])
acw_indices_3x3 = torch.tensor([0, 3, 6, 7, 8, 5, 2, 1, 4])

# Define reverse rearrangement indices for a 3x3 kernel
reverse_col_indices_3x3 = torch.tensor([0, 3, 6, 1, 4, 7, 2, 5, 8])
reverse_clw_indices_3x3 = torch.tensor([0, 1, 2, 7, 8, 3, 6, 5, 4])
reverse_acw_indices_3x3 = torch.tensor([0, 7, 6, 1, 8, 5, 2, 3, 4])

def rearrange_kernel_weights(weights, order):
    """
    Rearranges 4D convolutional weights to have specified-ordered kernels.
    Args:
        weights: Input tensor of shape [out_channels, in_channels, kernel_size, kernel_size]
        order: Indices to rearrange the 3x3 kernels
    Returns:
        Reshaped tensor with specified-ordered 3x3 kernels
    """
    out_channels, in_channels, h, w = weights.shape # h = w = self.kernel_size
    assert h == w == 3, "Only 3x3 kernels are supported (for now)"
    
    # Reshape to [out_channels * in_channels, 3, 3]
    kernels = weights.view(-1, h, w)
    
    # Process each kernel
    clockwise_kernels = []
    for kernel in kernels:
        flat_kernel = kernel.view(-1)
        clockwise_kernel = flat_kernel[order]
        clockwise_kernels.append(clockwise_kernel)
    
    # Stack and reshape back to [out_channels, in_channels, 9]
    clockwise_weights = torch.stack(clockwise_kernels)
    return clockwise_weights.view(out_channels, in_channels, h * w)