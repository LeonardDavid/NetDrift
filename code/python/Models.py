import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation

class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class VGG3(nn.Module):
    def __init__(self, quantMethod=None, quantize_train=True, quantize_eval=True, error_model=None, train_crit=None, test_crit=None, test_rtm = None, rt_error=0.0, global_rt_mapping = "MIX", kernel_size=3, rt_size=64, protectLayers=[], affected_rts=[], misalign_faults=[], bitflips=[], global_bitflip_budget=0.05, local_bitflip_budget=0.1, calc_results=True, calc_bitflips=True, calc_misalign_faults=True, calc_affected_rts=True):
        super(VGG3, self).__init__()
        self.htanh = nn.Hardtanh()
        self.name = "VGG3"
        self.quantization = quantMethod
        self.q_train = quantize_train
        self.q_test = quantize_eval
        self.error_model = error_model
        self.traincriterion = train_crit
        self.testcriterion = test_crit
        self.kernel_size = kernel_size

        self.test_rtm = test_rtm
        self.rt_error = rt_error
        self.rt_size = rt_size
        self.protectLayers = protectLayers
        self.affected_rts = affected_rts
        self.misalign_faults = misalign_faults
        self.bitflips = bitflips
        self.global_bitflip_budget = global_bitflip_budget
        self.local_bitflip_budget = local_bitflip_budget

        self.calc_results = calc_results
        self.calc_bitflips = calc_bitflips
        self.calc_misalign_fault = calc_misalign_faults
        self.calc_affected_rts = calc_affected_rts

        self.global_rt_mapping = global_rt_mapping

        self.initLayerDims()
        
        if self.test_rtm == 1:
            self.initIndexOffsets()
        else:
            self.initDefaultOffsets()

        self.conv1 = QuantizedConv2d(self.conv1_x, self.conv1_y, layerNr=1, rt_mapping=self.conv1_rt_mapping, rt_error=self.rt_error, protectLayers = self.protectLayers, affected_rts=self.affected_rts, padding=1, stride=1, quantization=self.quantization, error_model=self.error_model, test_rtm = self.test_rtm, kernel_size=self.kernel_size, index_offset = self.index_offset_conv1, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.qact1 = QuantizedActivation(quantization=self.quantization)

        self.conv2 = QuantizedConv2d(self.conv2_x, self.conv2_y, layerNr=2, rt_mapping=self.conv2_rt_mapping, rt_error=self.rt_error, protectLayers = self.protectLayers, affected_rts=self.affected_rts, padding=1, stride=1,  quantization=self.quantization, error_model=self.error_model, test_rtm = self.test_rtm, kernel_size=self.kernel_size, index_offset = self.index_offset_conv2, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.qact2 = QuantizedActivation(quantization=self.quantization)

        self.fc1 = QuantizedLinear(self.fc1_x, self.fc1_y, layerNr=3, rt_mapping=self.fc1_rt_mapping, rt_error=self.rt_error, protectLayers = self.protectLayers, affected_rts=self.affected_rts, quantization=self.quantization, error_model=self.error_model, test_rtm = self.test_rtm, index_offset = self.index_offset_fc1, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.bn3 = nn.BatchNorm1d(2048)
        self.qact3 = QuantizedActivation(quantization=self.quantization)

        self.fc2 = QuantizedLinear(self.fc2_x, self.fc2_y, layerNr=4, rt_mapping=self.fc2_rt_mapping, rt_error=self.rt_error, protectLayers = self.protectLayers, affected_rts=self.affected_rts, quantization=self.quantization, error_model=self.error_model, test_rtm = self.test_rtm, index_offset = self.index_offset_fc2, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.scale = Scale()


    def getRacetrackSize(self):
        return self.rt_size

    def initLayerDims(self):
        self.conv1_x = 1
        self.conv1_y = 64
        self.conv2_x = 64
        self.conv2_y = 64
        if self.kernel_size == 3:
            self.fc1_x = 7*7*64
        elif self.kernel_size == 5:
            self.fc1_x = 5*5*64
        elif self.kernel_size == 7:
            self.fc1_x = 4*4*64
        else:
            print("NO available FMNIST models for kernel size " + str(self.kernel_size))
            exit
        self.fc1_y = 2048
        self.fc2_x = 2048
        self.fc2_y = 10

        self.conv1_rt_mapping = None
        self.conv2_rt_mapping = None
        self.fc1_rt_mapping = None
        self.fc2_rt_mapping = None

    def initDefaultOffsets(self):
        self.index_offset_conv1 = np.zeros((1, 1), dtype=np.int32)
        self.index_offset_conv2 = np.zeros((1, 1), dtype=np.int32)
        self.index_offset_fc1 = np.zeros((1, 1), dtype=np.int32)
        self.index_offset_fc2 = np.zeros((1, 1), dtype=np.int32)
    
    def initIndexOffsets(self):
        if self.global_rt_mapping == "ROW":
            self.conv1_rt_mapping = "ROW"
            self.conv2_rt_mapping = "ROW"
            self.fc1_rt_mapping = "ROW"
            self.fc2_rt_mapping = "ROW"

        elif self.global_rt_mapping == "COL":
            self.conv1_rt_mapping = "COL"
            self.conv2_rt_mapping = "COL"
            self.fc1_rt_mapping = "COL"
            self.fc2_rt_mapping = "COL"

        elif self.global_rt_mapping == "MIX":
            if math.ceil((self.conv1_y/self.rt_size))*(self.conv1_x*self.kernel_size*self.kernel_size) <= math.ceil((self.conv1_x*self.kernel_size*self.kernel_size)/self.rt_size)*self.conv1_y:
                self.conv1_rt_mapping = "COL"
            else:
                self.conv1_rt_mapping = "ROW"
            
            if math.ceil((self.conv2_y/self.rt_size))*(self.conv2_x*self.kernel_size*self.kernel_size) <= math.ceil((self.conv2_x*self.kernel_size*self.kernel_size)/self.rt_size)*self.conv2_y:
                self.conv2_rt_mapping = "COL"
            else:
                self.conv2_rt_mapping = "ROW"

            if math.ceil((self.fc1_y/self.rt_size))*self.fc1_x <= math.ceil((self.fc1_x)/self.rt_size)*self.fc1_y:
                self.fc1_rt_mapping = "COL"
            else:
                self.fc1_rt_mapping = "ROW"
            
            if math.ceil((self.fc2_y/self.rt_size))*self.fc2_x <= math.ceil((self.fc2_x)/self.rt_size)*self.fc2_y:
                self.fc2_rt_mapping = "COL"
            else:
                self.fc2_rt_mapping = "ROW"
        else:
            print("No valid global_rt_mapping")
            exit()

        if self.conv1_rt_mapping == "ROW":
            io_conv1_x = self.conv1_y
            io_conv1_y = math.ceil((self.conv1_x * self.kernel_size * self.kernel_size)/self.rt_size)
        elif self.conv1_rt_mapping == "COL":
            io_conv1_x = self.conv1_x * self.kernel_size * self.kernel_size
            io_conv1_y = math.ceil(self.conv1_y/self.rt_size)

        if self.conv2_rt_mapping == "ROW":
            io_conv2_x = self.conv2_y
            io_conv2_y = math.ceil((self.conv2_x * self.kernel_size * self.kernel_size)/self.rt_size)
        elif self.conv2_rt_mapping == "COL":
            io_conv2_x = self.conv2_x * self.kernel_size * self.kernel_size
            io_conv2_y = math.ceil(self.conv2_y/self.rt_size)
        
        if self.fc1_rt_mapping == "ROW":
            io_fc1_x = self.fc1_y
            io_fc1_y = math.ceil(self.fc1_x/self.rt_size)
        elif self.fc1_rt_mapping == "COL":
            io_fc1_x = self.fc1_x
            io_fc1_y = math.ceil(self.fc1_y/self.rt_size)
            
        if self.fc2_rt_mapping == "ROW":
            io_fc2_x = self.fc2_y
            io_fc2_y = math.ceil(self.fc2_x/self.rt_size)
        elif self.fc2_rt_mapping == "COL":
            io_fc2_x = self.fc2_x
            io_fc2_y = math.ceil(self.fc2_y/self.rt_size)


        self.index_offset_conv1 = np.zeros((io_conv1_x, io_conv1_y), dtype=np.int32)
        self.index_offset_conv2 = np.zeros((io_conv2_x, io_conv2_y), dtype=np.int32)
        self.index_offset_fc1 = np.zeros((io_fc1_x, io_fc1_y), dtype=np.int32)
        self.index_offset_fc2 = np.zeros((io_fc2_x, io_fc2_y), dtype=np.int32)


    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.bn1(x)
        x = self.htanh(x)
        x = self.qact1(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.bn2(x)
        x = self.htanh(x)
        x = self.qact2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.htanh(x)
        x = self.qact2(x)

        x = self.fc2(x)
        x = self.scale(x)

        return x


class VGG7(nn.Module):
    def __init__(self, quantMethod=None, quantize_train=True, quantize_eval=True, error_model=None, train_crit=None, test_crit=None, test_rtm = None, rt_error=0.0,  global_rt_mapping = "MIX", kernel_size=3, rt_size=64, protectLayers=[], affected_rts=[], misalign_faults=[], bitflips=[], global_bitflip_budget=0.05, local_bitflip_budget=0.1, calc_results=True, calc_bitflips=True, calc_misalign_faults=True, calc_affected_rts=True):
        super(VGG7, self).__init__()
        self.htanh = nn.Hardtanh()
        self.name = "VGG7"
        self.quantization = quantMethod
        self.q_train = quantize_train
        self.q_test = quantize_eval
        self.error_model = error_model
        self.traincriterion = train_crit
        self.testcriterion = test_crit
        self.kernel_size = kernel_size

        self.test_rtm = test_rtm
        self.rt_error = rt_error
        self.rt_size = rt_size
        self.protectLayers = protectLayers
        self.affected_rts = affected_rts
        self.misalign_faults = misalign_faults
        self.bitflips = bitflips
        self.global_bitflip_budget = global_bitflip_budget
        self.local_bitflip_budget = local_bitflip_budget

        self.calc_results = calc_results
        self.calc_bitflips = calc_bitflips
        self.calc_misalign_fault = calc_misalign_faults
        self.calc_affected_rts = calc_affected_rts

        self.global_rt_mapping = global_rt_mapping

        self.initLayerDims()

        if self.test_rtm == 1:
            self.initIndexOffsets()
        else:
            self.initDefaultOffsets()

        #CNN
        # block 1
        self.conv1 = QuantizedConv2d(self.conv1_x, self.conv1_y, layerNr=1, rt_mapping=self.conv1_rt_mapping, rt_error=self.rt_error, protectLayers=self.protectLayers, affected_rts=self.affected_rts, kernel_size=self.kernel_size, padding=1, stride=1, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv1, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.qact1 = QuantizedActivation(quantization=self.quantization)

        # block 2
        self.conv2 = QuantizedConv2d(self.conv2_x, self.conv2_y, layerNr=2, rt_mapping=self.conv2_rt_mapping, rt_error=self.rt_error, protectLayers=self.protectLayers, affected_rts=self.affected_rts, kernel_size=self.kernel_size, padding=1, stride=1, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv2, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.qact2 = QuantizedActivation(quantization=self.quantization)

        # block 3
        self.conv3 = QuantizedConv2d(self.conv3_x, self.conv3_y, layerNr=3, rt_mapping=self.conv3_rt_mapping, rt_error=self.rt_error, protectLayers=self.protectLayers, affected_rts=self.affected_rts, kernel_size=self.kernel_size, padding=1, stride=1, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv3, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.qact3 = QuantizedActivation(quantization=self.quantization)

        # block 4
        self.conv4 = QuantizedConv2d(self.conv4_x, self.conv4_y, layerNr=4, rt_mapping=self.conv4_rt_mapping, rt_error=self.rt_error, protectLayers=self.protectLayers, affected_rts=self.affected_rts, kernel_size=self.kernel_size, padding=1, stride=1, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv4, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.qact4 = QuantizedActivation(quantization=self.quantization)

        # block 5
        self.conv5 = QuantizedConv2d(self.conv5_x, self.conv5_y, layerNr=5, rt_mapping=self.conv5_rt_mapping, rt_error=self.rt_error, protectLayers=self.protectLayers, affected_rts=self.affected_rts, kernel_size=self.kernel_size, padding=1, stride=1, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv5, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.qact5 = QuantizedActivation(quantization=self.quantization)

        # block 6
        self.conv6 = QuantizedConv2d(self.conv6_x, self.conv6_y, layerNr=6, rt_mapping=self.conv6_rt_mapping, rt_error=self.rt_error, protectLayers=self.protectLayers, affected_rts=self.affected_rts, kernel_size=self.kernel_size, padding=1, stride=1, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv6, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.qact6 = QuantizedActivation(quantization=self.quantization)

        # block 7
        self.fc1 = QuantizedLinear(self.fc1_x, self.fc1_y, layerNr=7, rt_mapping=self.fc1_rt_mapping, rt_error=self.rt_error, protectLayers=self.protectLayers, affected_rts=self.affected_rts, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_fc1, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_bitflips=calc_bitflips, calc_results=calc_results, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.bn7 = nn.BatchNorm1d(1024)
        self.qact7 = QuantizedActivation(quantization=self.quantization)

        self.fc2 = QuantizedLinear(self.fc2_x, self.fc2_y, layerNr=8, rt_mapping=self.fc2_rt_mapping, rt_error=self.rt_error, protectLayers=self.protectLayers, affected_rts=self.affected_rts, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_fc2, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_bitflips=calc_bitflips, calc_results=calc_results, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.scale = Scale(init_value=1e-3)


    def getRacetrackSize(self):
        return self.rt_size
    
    def initLayerDims(self):
        self.conv1_x = 3
        self.conv1_y = 128
        self.conv2_x = 128
        self.conv2_y = 128
        self.conv3_x = 128
        self.conv3_y = 256
        self.conv4_x = 256
        self.conv4_y = 256
        self.conv5_x = 256
        self.conv5_y = 512
        self.conv6_x = 512
        self.conv6_y = 512
        self.fc1_x = 8192
        self.fc1_y = 1024
        self.fc2_x = 1024
        self.fc2_y = 10

        self.conv1_rt_mapping = None
        self.conv2_rt_mapping = None
        self.conv3_rt_mapping = None
        self.conv4_rt_mapping = None
        self.conv5_rt_mapping = None
        self.conv6_rt_mapping = None
        self.fc1_rt_mapping = None
        self.fc2_rt_mapping = None

    def initDefaultOffsets(self):
        self.index_offset_conv1 = np.zeros((1, 1), dtype=np.int32)
        self.index_offset_conv2 = np.zeros((1, 1), dtype=np.int32)
        self.index_offset_conv3 = np.zeros((1, 1), dtype=np.int32)
        self.index_offset_conv4 = np.zeros((1, 1), dtype=np.int32)
        self.index_offset_conv5 = np.zeros((1, 1), dtype=np.int32)
        self.index_offset_conv6 = np.zeros((1, 1), dtype=np.int32)
        self.index_offset_fc1 = np.zeros((1, 1), dtype=np.int32)
        self.index_offset_fc2 = np.zeros((1, 1), dtype=np.int32)

    def initIndexOffsets(self):
        if self.global_rt_mapping == "ROW":
            self.conv1_rt_mapping = "ROW"
            self.conv2_rt_mapping = "ROW"
            self.conv3_rt_mapping = "ROW"
            self.conv4_rt_mapping = "ROW"
            self.conv5_rt_mapping = "ROW"
            self.conv6_rt_mapping = "ROW"
            self.fc1_rt_mapping = "ROW"
            self.fc2_rt_mapping = "ROW"

        elif self.global_rt_mapping == "COL":
            self.conv1_rt_mapping = "COL"
            self.conv2_rt_mapping = "COL"
            self.conv3_rt_mapping = "COL"
            self.conv4_rt_mapping = "COL"
            self.conv5_rt_mapping = "COL"
            self.conv6_rt_mapping = "COL"
            self.fc1_rt_mapping = "COL"
            self.fc2_rt_mapping = "COL"

        elif self.global_rt_mapping == "MIX":
            if math.ceil((self.conv1_y/self.rt_size))*(self.conv1_x*self.kernel_size*self.kernel_size) <= math.ceil((self.conv1_x*self.kernel_size*self.kernel_size)/self.rt_size)*self.conv1_y:
                self.conv1_rt_mapping = "COL"
            else:
                self.conv1_rt_mapping = "ROW"
            
            if math.ceil((self.conv2_y/self.rt_size))*(self.conv2_x*self.kernel_size*self.kernel_size) <= math.ceil((self.conv2_x*self.kernel_size*self.kernel_size)/self.rt_size)*self.conv2_y:
                self.conv2_rt_mapping = "COL"
            else:
                self.conv2_rt_mapping = "ROW"

            if math.ceil((self.conv3_y/self.rt_size))*(self.conv3_x*self.kernel_size*self.kernel_size) <= math.ceil((self.conv3_x*self.kernel_size*self.kernel_size)/self.rt_size)*self.conv3_y:
                self.conv3_rt_mapping = "COL"
            else:
                self.conv3_rt_mapping = "ROW"

            if math.ceil((self.conv4_y/self.rt_size))*(self.conv4_x*self.kernel_size*self.kernel_size) <= math.ceil((self.conv4_x*self.kernel_size*self.kernel_size)/self.rt_size)*self.conv4_y:
                self.conv4_rt_mapping = "COL"
            else:
                self.conv4_rt_mapping = "ROW"

            if math.ceil((self.conv5_y/self.rt_size))*(self.conv5_x*self.kernel_size*self.kernel_size) <= math.ceil((self.conv5_x*self.kernel_size*self.kernel_size)/self.rt_size)*self.conv5_y:
                self.conv5_rt_mapping = "COL"
            else:
                self.conv5_rt_mapping = "ROW"

            if math.ceil((self.conv6_y/self.rt_size))*(self.conv6_x*self.kernel_size*self.kernel_size) <= math.ceil((self.conv6_x*self.kernel_size*self.kernel_size)/self.rt_size)*self.conv6_y:
                self.conv6_rt_mapping = "COL"
            else:
                self.conv6_rt_mapping = "ROW"

            if math.ceil((self.fc1_y/self.rt_size))*self.fc1_x <= math.ceil((self.fc1_x)/self.rt_size)*self.fc1_y:
                self.fc1_rt_mapping = "COL"
            else:
                self.fc1_rt_mapping = "ROW"

            if math.ceil((self.fc2_y/self.rt_size))*self.fc2_x <= math.ceil((self.fc2_x)/self.rt_size)*self.fc2_y:
                self.fc2_rt_mapping = "COL"
            else:
                self.fc2_rt_mapping = "ROW"
        else:
            print("No valid global_rt_mapping")
            exit()

        if self.conv1_rt_mapping == "ROW":
            io_conv1_x = self.conv1_y
            io_conv1_y = math.ceil((self.conv1_x * self.kernel_size * self.kernel_size)/self.rt_size)
        elif self.conv1_rt_mapping == "COL":
            io_conv1_x = self.conv1_x * self.kernel_size * self.kernel_size
            io_conv1_y = math.ceil(self.conv1_y/self.rt_size)

        if self.conv2_rt_mapping == "ROW":
            io_conv2_x = self.conv2_y
            io_conv2_y = math.ceil((self.conv2_x * self.kernel_size * self.kernel_size)/self.rt_size)
        elif self.conv2_rt_mapping == "COL":
            io_conv2_x = self.conv2_x * self.kernel_size * self.kernel_size
            io_conv2_y = math.ceil(self.conv2_y/self.rt_size)

        if self.conv3_rt_mapping == "ROW":
            io_conv3_x = self.conv3_y
            io_conv3_y = math.ceil((self.conv3_x * self.kernel_size * self.kernel_size)/self.rt_size)
        elif self.conv3_rt_mapping == "COL":
            io_conv3_x = self.conv3_x * self.kernel_size * self.kernel_size
            io_conv3_y = math.ceil(self.conv3_y/self.rt_size)

        if self.conv4_rt_mapping == "ROW":
            io_conv4_x = self.conv4_y
            io_conv4_y = math.ceil((self.conv4_x * self.kernel_size * self.kernel_size)/self.rt_size)
        elif self.conv4_rt_mapping == "COL":
            io_conv4_x = self.conv4_x * self.kernel_size * self.kernel_size
            io_conv4_y = math.ceil(self.conv4_y/self.rt_size)

        if self.conv5_rt_mapping == "ROW":
            io_conv5_x = self.conv5_y
            io_conv5_y = math.ceil((self.conv5_x * self.kernel_size * self.kernel_size)/self.rt_size)
        elif self.conv5_rt_mapping == "COL":
            io_conv5_x = self.conv5_x * self.kernel_size * self.kernel_size
            io_conv5_y = math.ceil(self.conv5_y/self.rt_size)

        if self.conv6_rt_mapping == "ROW":
            io_conv6_x = self.conv6_y
            io_conv6_y = math.ceil((self.conv6_x * self.kernel_size * self.kernel_size)/self.rt_size)
        elif self.conv6_rt_mapping == "COL":
            io_conv6_x = self.conv6_x * self.kernel_size * self.kernel_size
            io_conv6_y = math.ceil(self.conv6_y/self.rt_size)

        if self.fc1_rt_mapping == "ROW":
            io_fc1_x = self.fc1_y
            io_fc1_y = math.ceil(self.fc1_x/self.rt_size)
        elif self.fc1_rt_mapping == "COL": 
            io_fc1_x = self.fc1_x
            io_fc1_y = math.ceil(self.fc1_y/self.rt_size)

        if self.fc2_rt_mapping == "ROW":
            io_fc2_x = self.fc2_y
            io_fc2_y = math.ceil(self.fc2_x/self.rt_size)
        elif self.fc2_rt_mapping == "COL":
            io_fc2_x = self.fc2_x
            io_fc2_y = math.ceil(self.fc2_y/self.rt_size)

        self.index_offset_conv1 = np.zeros((io_conv1_x, io_conv1_y), dtype=np.int32)
        self.index_offset_conv2 = np.zeros((io_conv2_x, io_conv2_y), dtype=np.int32)
        self.index_offset_conv3 = np.zeros((io_conv3_x, io_conv3_y), dtype=np.int32)
        self.index_offset_conv4 = np.zeros((io_conv4_x, io_conv4_y), dtype=np.int32)
        self.index_offset_conv5 = np.zeros((io_conv5_x, io_conv5_y), dtype=np.int32)
        self.index_offset_conv6 = np.zeros((io_conv6_x, io_conv6_y), dtype=np.int32)
        self.index_offset_fc1 = np.zeros((io_fc1_x, io_fc1_y), dtype=np.int32)
        self.index_offset_fc2 = np.zeros((io_fc2_x, io_fc2_y), dtype=np.int32)
    

    def forward(self, x):
        # block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.htanh(x)
        x = self.qact1(x)

        # block 2
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.bn2(x)
        x = self.htanh(x)
        x = self.qact2(x)

        # block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.htanh(x)
        x = self.qact3(x)

        # block 4
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        x = self.bn4(x)
        x = self.htanh(x)
        x = self.qact4(x)

        # block 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.htanh(x)
        x = self.qact5(x)

        # block 6
        x = self.conv6(x)
        x = F.max_pool2d(x, 2)
        x = self.bn6(x)
        x = self.htanh(x)
        x = self.qact6(x)

        # block 7
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn7(x)
        x = self.htanh(x)
        x = self.qact3(x)

        x = self.fc2(x)
        x = self.scale(x)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, layerNr=2, test_rtm = None, rt_size=64, global_rt_mapping="MIX", rt_error=0.0, protectLayers=[], affected_rts=[], misalign_faults=[], bitflips=[], global_bitflip_budget=0.05, local_bitflip_budget=0.1, calc_results=True, calc_bitflips=True, calc_misalign_faults=True, calc_affected_rts=True, quantMethod=None, an_sim=None, array_size=None, mapping=None, mapping_distr=None, sorted_mapping_idx=None, performance_mode=None, error_model=None, train_model=None, extract_absfreq=None):
        super(BasicBlock, self).__init__()
        self.htanh = nn.Hardtanh()
        self.layerNr = layerNr
        self.kernel_conv1 = 3
        self.kernel_conv2 = 3
        self.kernel_short = 1
        self.qact = QuantizedActivation(quantization=quantMethod)

        self.test_rtm = test_rtm
        self.rt_error = rt_error
        self.rt_size = rt_size
        self.protectLayers = protectLayers
        self.affected_rts = affected_rts
        self.misalign_faults = misalign_faults
        self.bitflips = bitflips
        self.global_bitflip_budget = global_bitflip_budget
        self.local_bitflip_budget = local_bitflip_budget

        self.calc_results = calc_results
        self.calc_bitflips = calc_bitflips
        self.calc_misalign_fault = calc_misalign_faults
        self.calc_affected_rts = calc_affected_rts

        self.global_rt_mapping = global_rt_mapping

        self.initLayerDims(in_planes, planes)

        if self.test_rtm == 1:
            self.initIndexOffsets()
        else:
            self.initDefaultOffsets()

        self.conv1 = QuantizedConv2d(
            self.conv1_x, self.conv1_y, layerNr=self.layerNr, stride=stride, padding=1, quantization=quantMethod, kernel_size=self.kernel_conv1, rt_mapping=self.conv1_rt_mapping, rt_error=self.rt_error, protectLayers=protectLayers, affected_rts=self.affected_rts, test_rtm = test_rtm, index_offset = self.index_offset_conv1, rt_size = self.rt_size, error_model=error_model, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, train_model=train_model, extract_absfreq=extract_absfreq, an_sim=an_sim, array_size=array_size, mac_mapping=mapping, mac_mapping_distr=mapping_distr, sorted_mac_mapping_idx=sorted_mapping_idx, performance_mode=performance_mode, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.layerNr += 1

        self.conv2 = QuantizedConv2d(
            self.conv2_x, self.conv2_y, layerNr=self.layerNr, stride=1, padding=1, quantization=quantMethod, kernel_size=self.kernel_conv2, rt_mapping=self.conv2_rt_mapping, rt_error=self.rt_error, protectLayers=protectLayers, affected_rts=self.affected_rts, test_rtm = test_rtm, index_offset=self.index_offset_conv2, rt_size = self.rt_size, error_model=error_model, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, train_model=train_model, extract_absfreq=extract_absfreq, an_sim=an_sim, array_size=array_size, mac_mapping=mapping, mac_mapping_distr=mapping_distr, sorted_mac_mapping_idx=sorted_mapping_idx, performance_mode=performance_mode,  bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.layerNr += 1

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            
            self.shortcut = nn.Sequential(
                QuantizedConv2d(
                    self.shortcut_x, self.shortcut_y, layerNr=self.layerNr, stride=stride, quantization=quantMethod, kernel_size=self.kernel_short, rt_mapping = self.shortcut_rt_mapping, rt_error=self.rt_error, protectLayers=protectLayers, affected_rts=self.affected_rts, test_rtm = test_rtm, index_offset = self.index_offset_shortcut, rt_size = self.rt_size, error_model=error_model, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, train_model=train_model, extract_absfreq=extract_absfreq, an_sim=an_sim, array_size=array_size, mac_mapping=mapping, mac_mapping_distr=mapping_distr, sorted_mac_mapping_idx=sorted_mapping_idx, performance_mode=performance_mode,                         bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
            self.layerNr += 1


    def getRacetrackSize(self):
        return self.rt_size

    def getLayerNr(self):
        return self.layerNr
    
    def initLayerDims(self, in_planes, planes):
        self.conv1_x = in_planes
        self.conv1_y = planes
        self.conv2_x = planes
        self.conv2_y = planes
        self.shortcut_x = in_planes
        self.shortcut_y = self.expansion*planes

        self.conv1_rt_mapping = None
        self.conv2_rt_mapping = None
        self.shortcut_rt_mapping = None

    def initDefaultOffsets(self):
        self.index_offset_conv1 = np.zeros((1, 1), dtype=np.int32)
        self.index_offset_conv2 = np.zeros((1, 1), dtype=np.int32)
        self.index_offset_shortcut = np.zeros((1, 1), dtype=np.int32)

    def initIndexOffsets(self):
        if self.global_rt_mapping == "ROW":
            self.conv1_rt_mapping = "ROW"
            self.conv2_rt_mapping = "ROW"
            self.shortcut_rt_mapping = "ROW"
        elif self.global_rt_mapping == "COL":
            self.conv1_rt_mapping = "COL"
            self.conv2_rt_mapping = "COL"
            self.shortcut_rt_mapping = "COL"
        elif self.global_rt_mapping == "MIX":
            if math.ceil((self.conv1_y/self.rt_size))*(self.conv1_x*self.kernel_conv1*self.kernel_conv1) <= math.ceil((self.conv1_x*self.kernel_conv1*self.kernel_conv1)/self.rt_size)*self.conv1_y:
                self.conv1_rt_mapping = "COL"
            else:
                self.conv1_rt_mapping = "ROW"
            
            if math.ceil((self.conv2_y/self.rt_size))*(self.conv2_x*self.kernel_conv2*self.kernel_conv2) <= math.ceil((self.conv2_x*self.kernel_conv2*self.kernel_conv2)/self.rt_size)*self.conv2_y:
                self.conv2_rt_mapping = "COL"
            else:
                self.conv2_rt_mapping = "ROW"

            if math.ceil((self.shortcut_y/self.rt_size))*(self.shortcut_x*self.kernel_short*self.kernel_short) <= math.ceil((self.shortcut_x*self.kernel_short*self.kernel_short)/self.rt_size)*self.shortcut_y:
                self.shortcut_rt_mapping = "COL"
            else:
                self.shortcut_rt_mapping = "ROW"
        else:
            print("No valid global_rt_mapping")
            exit()

        if self.conv1_rt_mapping == "ROW":
            io_conv1_x = self.conv1_y
            io_conv1_y = math.ceil((self.conv1_x * self.kernel_conv1 * self.kernel_conv1)/self.rt_size)
        elif self.conv1_rt_mapping == "COL":
            io_conv1_x = self.conv1_x * self.kernel_conv1 * self.kernel_conv1
            io_conv1_y = math.ceil(self.conv1_y/self.rt_size)

        if self.conv2_rt_mapping == "ROW":
            io_conv2_x = self.conv2_y
            io_conv2_y = math.ceil((self.conv2_x * self.kernel_conv2 * self.kernel_conv2)/self.rt_size)
        elif self.conv2_rt_mapping == "COL":
            io_conv2_x = self.conv2_x * self.kernel_conv2 * self.kernel_conv2
            io_conv2_y = math.ceil(self.conv2_y/self.rt_size)

        if self.shortcut_rt_mapping == "ROW":
            io_shortcut_x = self.shortcut_y
            io_shortcut_y = math.ceil((self.shortcut_x * self.kernel_short * self.kernel_short)/self.rt_size)
        elif self.shortcut_rt_mapping == "COL":
            io_shortcut_x = self.shortcut_x * self.kernel_short * self.kernel_short
            io_shortcut_y = math.ceil(self.shortcut_y/self.rt_size)

        self.index_offset_conv1 = np.zeros((io_conv1_x, io_conv1_y), dtype=np.int32)
        self.index_offset_conv2 = np.zeros((io_conv2_x, io_conv2_y), dtype=np.int32)
        self.index_offset_shortcut = np.zeros((io_shortcut_x, io_shortcut_y), dtype=np.int32)


    def forward(self, x):
        out = self.qact(self.htanh(self.bn1(self.conv1(x))))
        # out = self.qact(self.htanh(self.bn2(self.conv2(out))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.qact(self.htanh(out))
        return out
    

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, train_crit, test_crit, quantMethod=None,  quantize_train=True, quantize_eval=True, error_model=None, an_sim=None, array_size=None, mapping=None, mapping_distr=None, sorted_mapping_idx=None, performance_mode=None, train_model=None, extract_absfreq=None, num_classes=10, test_rtm = None, rt_error=0.0, global_rt_mapping = "MIX", kernel_size=3, rt_size=64, protectLayers=[], affected_rts=[], misalign_faults=[], bitflips=[], global_bitflip_budget=0.05, local_bitflip_budget=0.1, calc_results=True, calc_bitflips=True, calc_misalign_faults=True, calc_affected_rts=True):
        super(ResNet, self).__init__()
        self.htanh = nn.Hardtanh()
        self.name = "ResNet18"
        self.quantization = quantMethod
        self.q_train = quantize_train
        self.q_test = quantize_eval
        self.error_model = error_model
        self.traincriterion = train_crit
        self.testcriterion = test_crit
        self.an_sim = an_sim
        self.array_size = array_size
        self.mapping = mapping
        self.mapping_distr = mapping_distr
        self.sorted_mapping_idx = sorted_mapping_idx
        self.performance_mode = performance_mode
        self.train_model = train_model
        self.extract_absfreq = extract_absfreq
        self.in_planes = 64
        self.kernel_size = kernel_size
        self.qact = QuantizedActivation(quantization=self.quantization)
        
        self.test_rtm = test_rtm
        self.rt_error = rt_error
        self.rt_size = rt_size
        self.protectLayers = protectLayers
        self.affected_rts = affected_rts
        self.misalign_faults = misalign_faults
        self.bitflips = bitflips
        self.global_bitflip_budget = global_bitflip_budget
        self.local_bitflip_budget = local_bitflip_budget

        self.calc_results = calc_results
        self.calc_bitflips = calc_bitflips
        self.calc_misalign_fault = calc_misalign_faults
        self.calc_affected_rts = calc_affected_rts

        self.global_rt_mapping = global_rt_mapping

        self.initLayerDims(block, num_classes)

        if self.test_rtm == 1:
            self.initIndexOffsets()
        else:
            self.initDefaultOffsets()

        self.layerNr = 1
        self.conv1 = QuantizedConv2d(
            self.conv1_x, self.conv1_y, affected_rts=self.affected_rts, layerNr=self.layerNr, rt_mapping=self.conv1_rt_mapping, rt_error=self.rt_error, protectLayers = self.protectLayers, kernel_size=self.kernel_size, stride=1, padding=1, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv1, rt_size = self.rt_size, bias=False, array_size=self.array_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts)
        self.bn1 = nn.BatchNorm2d(64)

        self.layerNr += 1
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = QuantizedLinear(
            self.lin1_x, self.lin1_y, affected_rts=self.affected_rts, layerNr=self.layerNr, rt_mapping=self.lin1_rt_mapping, rt_error=self.rt_error, protectLayers=self.protectLayers, quantization=self.quantization, an_sim=self.an_sim, array_size=self.array_size, mac_mapping=self.mapping, mac_mapping_distr=self.mapping_distr, sorted_mac_mapping_idx=self.sorted_mapping_idx, performance_mode=self.performance_mode, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_lin1, rt_size = self.rt_size, bias=False, train_model=self.train_model, extract_absfreq=self.extract_absfreq, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:

            bblock = block(
                self.in_planes, planes, stride, layerNr = self.layerNr, test_rtm = self.test_rtm, rt_size=self.rt_size, protectLayers=self.protectLayers, affected_rts=self.affected_rts, quantMethod=self.quantization, an_sim=self.an_sim, array_size=self.array_size, mapping=self.mapping, mapping_distr=self.mapping_distr, sorted_mapping_idx=self.sorted_mapping_idx, performance_mode=self.performance_mode, error_model=self.error_model, train_model=self.train_model, extract_absfreq=self.extract_absfreq)

            layers.append(bblock)
            self.in_planes = planes * block.expansion
            self.layerNr = bblock.getLayerNr()

        return nn.Sequential(*layers)
    
    
    def getRacetrackSize(self):
        return self.rt_size
    
    def initLayerDims(self, block, num_classes):
        self.conv1_x = 3
        self.conv1_y = 64
        self.lin1_x = 512*block.expansion
        self.lin1_y = num_classes

        self.conv1_rt_mapping = None
        self.lin1_rt_mapping = None
    
    def initDefaultOffsets(self):
        self.index_offset_conv1 = np.zeros((1, 1), dtype=np.int32)
        self.index_offset_lin1 = np.zeros((1, 1), dtype=np.int32)

    def initIndexOffsets(self):
        print("global_rt_mapping: ", self.global_rt_mapping)
        if self.global_rt_mapping == "ROW":
            self.conv1_rt_mapping = "ROW"
            self.lin1_rt_mapping = "ROW"

        elif self.global_rt_mapping == "COL":
            self.conv1_rt_mapping = "COL"
            self.lin1_rt_mapping = "COL"

        elif self.global_rt_mapping == "MIX":
            if math.ceil((self.conv1_y/self.rt_size))*(self.conv1_x*self.kernel_size*self.kernel_size) <= math.ceil((self.conv1_x*self.kernel_size*self.kernel_size)/self.rt_size)*self.conv1_y:
                self.conv1_rt_mapping = "COL"
            else:
                self.conv1_rt_mapping = "ROW"

            if math.ceil((self.lin1_y/self.rt_size))*self.lin1_x <= math.ceil((self.lin1_x)/self.rt_size)*self.lin1_y:
                self.lin1_rt_mapping = "COL"
            else:
                self.lin1_rt_mapping = "ROW"
        else:
            print("No valid global_rt_mapping")
            exit()

        if self.conv1_rt_mapping == "ROW":
            io_conv1_x = self.conv1_y
            io_conv1_y = math.ceil((self.conv1_x * self.kernel_size * self.kernel_size)/self.rt_size)
        elif self.conv1_rt_mapping == "COL":
            io_conv1_x = self.conv1_x * self.kernel_size * self.kernel_size
            io_conv1_y = math.ceil(self.conv1_y/self.rt_size)

        if self.lin1_rt_mapping == "ROW":
            io_lin1_x = self.lin1_y
            io_lin1_y = math.ceil(self.lin1_x/self.rt_size)
        elif self.lin1_rt_mapping == "COL":
            io_lin1_x = self.lin1_x
            io_lin1_y = math.ceil(self.lin1_y/self.rt_size)

        self.index_offset_conv1 = np.zeros((io_conv1_x, io_conv1_y), dtype=np.int32)
        self.index_offset_lin1 = np.zeros((io_lin1_x, io_lin1_y), dtype=np.int32)


    def forward(self, x):
        out = self.qact(self.htanh(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.max_pool2d(out, 2)
        out = self.layer4(out)
        out = F.max_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


### MLP ###
class MLP(nn.Module):
    def __init__(self, quantMethod=None, quantize_train=True, quantize_eval=True, error_model=None, test_rtm = None, rt_size=64, protectLayers=[], affected_rts=[], misalign_faults=[], bitflips=[], global_bitflip_budget=0.05, local_bitflip_budget=0.1, calc_results=True, calc_bitflips=True, calc_misalign_faults=True, calc_affected_rts=True):
        super(MLP, self).__init__()
        self.name = "MLP"
        self.quantization = quantMethod
        self.q_train = quantize_train
        self.q_test = quantize_eval
        self.error_model = error_model
        self.htanh = nn.Hardtanh()
        self.rt_size = rt_size
        self.protectLayers = protectLayers
        self.affected_rts = affected_rts
        self.misalign_faults = misalign_faults
        self.bitflips = bitflips
        self.global_bitflip_budget = global_bitflip_budget
        self.local_bitflip_budget = local_bitflip_budget

        self.calc_results = calc_results
        self.calc_bitflips = calc_bitflips
        self.calc_misalign_fault = calc_misalign_faults
        self.calc_affected_rts = calc_affected_rts
        
        ### FP ###
        # # number of hidden nodes in each layer (512)
        # # linear layer (784 -> hidden_1)
        # self.fc1 = nn.Linear(28 * 28, 512)
        # # linear layer (n_hidden -> hidden_2)
        # self.fc2 = nn.Linear(512, 512)
        # # linear layer (n_hidden -> 10)
        # self.fc3 = nn.Linear(512, 10)
        # # dropout layer (p=0.2)
        # # dropout prevents overfitting of data
        # self.dropout = nn.Dropout(0.2)


        self.resetOffsets()

        ### 9696 BNN ###
        # TODO bias = True?

        # self.fc1 = QuantizedLinear(28*28, 512, layerNr=1, protectLayers = self.protectLayers, affected_rts=self.affected_rts, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_fc1, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.qact1 = QuantizedActivation(quantization=self.quantization)

        # self.fc2 = QuantizedLinear(512, 512, layerNr=2, protectLayers = self.protectLayers, affected_rts=self.affected_rts, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_fc2, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        # self.bn2 = nn.BatchNorm1d(512)
        # self.qact2 = QuantizedActivation(quantization=self.quantization)

        # self.fc3 = QuantizedLinear(512, 10, layerNr=3, protectLayers = self.protectLayers, affected_rts=self.affected_rts, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_fc3, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        # # self.scale = Scale()
        # self.dropout = nn.Dropout(0.2)


        ### 9418 BNN ###
        # TODO bias = True?

        self.fc1 = QuantizedLinear(28*28, 512, layerNr=1, protectLayers = self.protectLayers, affected_rts=self.affected_rts, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_fc1, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.qact1 = QuantizedActivation(quantization=self.quantization)

        # self.fc2 = QuantizedLinear(512, 512, layerNr=2, protectLayers = self.protectLayers, affected_rts=self.affected_rts, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_fc2, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        # self.bn2 = nn.BatchNorm1d(512)
        # self.qact2 = QuantizedActivation(quantization=self.quantization)

        self.fc3 = QuantizedLinear(512, 10, layerNr=2, protectLayers = self.protectLayers, affected_rts=self.affected_rts, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_fc3, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        # self.scale = Scale()
        self.dropout = nn.Dropout(0.2)


        ### xxx BNN ###
        # TODO bias = True?

        # self.fc1 = QuantizedLinear(28*28, 512, layerNr=1, protectLayers = self.protectLayers, affected_rts=self.affected_rts, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_fc1, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.qact1 = QuantizedActivation(quantization=self.quantization)

        # self.fc2 = QuantizedLinear(512, 512, layerNr=2, protectLayers = self.protectLayers, affected_rts=self.affected_rts, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_fc2, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        # self.bn2 = nn.BatchNorm1d(512)
        # self.qact2 = QuantizedActivation(quantization=self.quantization)

        self.fc3 = QuantizedLinear(28*28, 10, layerNr=1, protectLayers = self.protectLayers, affected_rts=self.affected_rts, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_fc3, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        # self.scale = Scale()
        self.dropout = nn.Dropout(0.2)
        

    def resetOffsets(self):

        if self.rt_size > 28*28: 
            fc1_y = 1
        else:
            fc1_y = int(28*28/self.rt_size+1) # +1 bc 784%64!=0
        self.index_offset_fc1 = np.zeros((512, fc1_y))

        if self.rt_size > 512: 
            fc2_y = 1
        else:
            fc2_y = int(512/self.rt_size)
        self.index_offset_fc2 = np.zeros((512, fc2_y))
        
        if self.rt_size > 2048: 
            fc3_y = 1
        else:
            fc3_y = int(2048/self.rt_size)
        self.index_offset_fc3 = np.zeros((10, fc3_y))


    def forward(self, x):

        ### FP ###
        # # flatten image input
        # x = x.view(-1, 28 * 28)

        # # add hidden layer, with relu activation function
        # x = F.relu(self.fc1(x))
        # # add dropout layer
        # x = self.dropout(x)

        # # add hidden layer, with relu activation function
        # x = F.relu(self.fc2(x))
        # # add dropout layer
        # x = self.dropout(x)

        # # add output layer
        # x = self.fc3(x)


        ### 9696 BNN ###
        # # flatten image input
        # x = x.view(-1, 28 * 28)

        # x = self.fc1(x)
        # x = self.bn1(x)
        # x = self.htanh(x)
        # x = self.qact1(x)
        # # add dropout layer
        # x = self.dropout(x)


        # x = self.fc2(x)
        # x = self.bn2(x)
        # x = self.htanh(x)
        # x = self.qact2(x)
        # # add dropout layer
        # x = self.dropout(x)

        # x = self.fc3(x)


        ### 9418 BNN ###
        # # flatten image input
        # x = x.view(-1, 28 * 28)

        # x = self.fc1(x)
        # x = self.bn1(x)
        # x = self.htanh(x)
        # x = self.qact1(x)
        # # add dropout layer
        # x = self.dropout(x)


        # x = self.fc2(x)
        # x = self.bn2(x)
        # x = self.htanh(x)
        # x = self.qact2(x)
        # # add dropout layer
        # x = self.dropout(x)

        # x = self.fc3(x)


        ### xxx BNN ###
        # flatten image input
        x = x.view(-1, 28 * 28)

        # x = self.fc1(x)
        # x = self.bn1(x)
        # x = self.htanh(x)
        # x = self.qact1(x)
        # # add dropout layer
        # x = self.dropout(x)


        # x = self.fc2(x)
        # x = self.bn2(x)
        # x = self.htanh(x)
        # x = self.qact2(x)
        # # add dropout layer
        # x = self.dropout(x)

        x = self.fc3(x)


        return x
    
