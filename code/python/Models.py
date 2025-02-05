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
    def __init__(self, quantMethod=None, quantize_train=True, quantize_eval=True, error_model=None, train_crit=None, test_crit=None, test_rtm = None, kernel_size=3, rt_size=64, protectLayers=[], affected_rts=[], misalign_faults=[], bitflips=[], global_bitflip_budget=0.05, local_bitflip_budget=0.1, calc_results=True, calc_bitflips=True, calc_misalign_faults=True, calc_affected_rts=True):
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

        self.resetOffsets()

        self.conv1 = QuantizedConv2d(self.conv1_x, self.conv1_y, layerNr=1, protectLayers = self.protectLayers, affected_rts=self.affected_rts, padding=1, stride=1, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, kernel_size=self.kernel_size, index_offset = self.index_offset_conv1, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.qact1 = QuantizedActivation(quantization=self.quantization)

        self.conv2 = QuantizedConv2d(self.conv2_x, self.conv2_y, layerNr=2, protectLayers = self.protectLayers, affected_rts=self.affected_rts, padding=1, stride=1,  quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, kernel_size=self.kernel_size, index_offset = self.index_offset_conv2, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.qact2 = QuantizedActivation(quantization=self.quantization)

        self.fc1 = QuantizedLinear(self.fc1_x, self.fc1_y, layerNr=3, protectLayers = self.protectLayers, affected_rts=self.affected_rts, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_fc1, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.bn3 = nn.BatchNorm1d(2048)
        self.qact3 = QuantizedActivation(quantization=self.quantization)

        self.fc2 = QuantizedLinear(self.fc2_x, self.fc2_y, layerNr=4, protectLayers = self.protectLayers, affected_rts=self.affected_rts, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_fc2, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.scale = Scale()
        
    
    def resetOffsets(self):

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

        self.index_offset_conv1 = np.zeros((math.ceil(self.conv1_x*self.conv2_x/self.rt_size), self.kernel_size * self.kernel_size))
        self.index_offset_conv2 = np.zeros((math.ceil(self.conv2_x*self.conv2_y/self.rt_size), self.kernel_size * self.kernel_size))
        self.index_offset_fc1 = np.zeros((math.ceil(self.fc1_x*self.fc1_y/self.rt_size), 1))
        self.index_offset_fc2 = np.zeros((math.ceil(self.fc2_x*self.fc2_y/self.rt_size), 1))

    
    def getRacetrackSize(self):
        return self.rt_size


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
    def __init__(self, quantMethod=None, quantize_train=True, quantize_eval=True, error_model=None, train_crit=None, test_crit=None, test_rtm = None, kernel_size=3, rt_size=64, protectLayers=[], affected_rts=[], misalign_faults=[], bitflips=[], global_bitflip_budget=0.05, local_bitflip_budget=0.1, calc_results=True, calc_bitflips=True, calc_misalign_faults=True, calc_affected_rts=True):
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

        self.resetOffsets()

        #CNN
        # block 1
        self.conv1 = QuantizedConv2d(self.conv1_x, self.conv1_y, protectLayers=self.protectLayers, affected_rts=self.affected_rts, kernel_size=self.kernel_size, padding=1, stride=1, quantization=self.quantization, layerNr=1, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv1, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.qact1 = QuantizedActivation(quantization=self.quantization)

        # block 2
        self.conv2 = QuantizedConv2d(self.conv2_x, self.conv2_y, protectLayers=self.protectLayers, affected_rts=self.affected_rts, kernel_size=self.kernel_size, padding=1, stride=1, quantization=self.quantization, layerNr=2, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv2, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.qact2 = QuantizedActivation(quantization=self.quantization)

        # block 3
        self.conv3 = QuantizedConv2d(self.conv3_x, self.conv3_y, protectLayers=self.protectLayers, affected_rts=self.affected_rts, kernel_size=self.kernel_size, padding=1, stride=1, quantization=self.quantization, layerNr=3, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv3, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.qact3 = QuantizedActivation(quantization=self.quantization)

        # block 4
        self.conv4 = QuantizedConv2d(self.conv4_x, self.conv4_y, protectLayers=self.protectLayers, affected_rts=self.affected_rts, kernel_size=self.kernel_size, padding=1, stride=1, quantization=self.quantization, layerNr=4, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv4, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.qact4 = QuantizedActivation(quantization=self.quantization)

        # block 5
        self.conv5 = QuantizedConv2d(self.conv5_x, self.conv5_y, protectLayers=self.protectLayers, affected_rts=self.affected_rts, kernel_size=self.kernel_size, padding=1, stride=1, quantization=self.quantization, layerNr=5, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv5, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.qact5 = QuantizedActivation(quantization=self.quantization)

        # block 6
        self.conv6 = QuantizedConv2d(self.conv6_x, self.conv6_y, protectLayers=self.protectLayers, affected_rts=self.affected_rts, kernel_size=self.kernel_size, padding=1, stride=1, quantization=self.quantization, layerNr=6, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv6, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.qact6 = QuantizedActivation(quantization=self.quantization)

        # block 7
        self.fc1 = QuantizedLinear(self.fc1_x, self.fc1_y, protectLayers=self.protectLayers, affected_rts=self.affected_rts, quantization=self.quantization, layerNr=7, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_fc1, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_bitflips=calc_bitflips, calc_results=calc_results, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.bn7 = nn.BatchNorm1d(1024)
        self.qact7 = QuantizedActivation(quantization=self.quantization)

        self.fc2 = QuantizedLinear(self.fc2_x, self.fc2_y, protectLayers=self.protectLayers, affected_rts=self.affected_rts, quantization=self.quantization, layerNr=8, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_fc2, rt_size = self.rt_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_bitflips=calc_bitflips, calc_results=calc_results, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts, bias=False)
        self.scale = Scale(init_value=1e-3)

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
 
    def getRacetrackSize(self):
        return self.rt_size

    def resetOffsets(self):

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

        self.index_offset_conv1 = np.zeros((math.ceil(self.conv1_x*self.conv1_y/self.rt_size), self.kernel_size * self.kernel_size))
        self.index_offset_conv2 = np.zeros((math.ceil(self.conv2_x*self.conv2_y/self.rt_size), self.kernel_size * self.kernel_size))
        self.index_offset_conv3 = np.zeros((math.ceil(self.conv3_x*self.conv3_y/self.rt_size), self.kernel_size * self.kernel_size))
        self.index_offset_conv4 = np.zeros((math.ceil(self.conv4_x*self.conv4_y/self.rt_size), self.kernel_size * self.kernel_size))
        self.index_offset_conv5 = np.zeros((math.ceil(self.conv5_x*self.conv5_y/self.rt_size), self.kernel_size * self.kernel_size))
        self.index_offset_conv6 = np.zeros((math.ceil(self.conv6_x*self.conv6_y/self.rt_size), self.kernel_size * self.kernel_size))
        self.index_offset_fc1 = np.zeros((math.ceil(self.fc1_x*self.fc1_y/self.rt_size), 1))
        self.index_offset_fc2 = np.zeros((math.ceil(self.fc2_x*self.fc2_y/self.rt_size), 1))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, kernel_size=3, quantMethod=None, an_sim=None, array_size=None, mapping=None, mapping_distr=None, sorted_mapping_idx=None, performance_mode=None, quantize_train=True, quantize_eval=True, error_model=None, train_model=None, extract_absfreq=None, test_rtm = None, rt_size=64, layerNr=2, protectLayers=[], affected_rts=[]):
        super(BasicBlock, self).__init__()
        self.htanh = nn.Hardtanh()
        self.kernel_size = kernel_size
        self.rt_size = rt_size
        self.layerNr = layerNr
        self.protectLayers = protectLayers
        self.affected_rts = affected_rts
        # print("##BLOCK##")
    
        self.qact = QuantizedActivation(quantization=quantMethod)

        self.resetConvOffsets(in_planes, planes)

        self.conv1 = QuantizedConv2d(
            self.conv1_x, self.conv1_y, affected_rts=self.affected_rts, layerNr=self.layerNr, protectLayers=protectLayers, kernel_size=3, stride=stride, padding=1, quantization=quantMethod, an_sim=an_sim, array_size=array_size, mac_mapping=mapping, mac_mapping_distr=mapping_distr, sorted_mac_mapping_idx=sorted_mapping_idx,
            performance_mode=performance_mode, test_rtm = test_rtm, index_offset = self.index_offset_conv1, rt_size = self.rt_size,
            error_model=error_model, bias=False, train_model=train_model, extract_absfreq=extract_absfreq)
        self.bn1 = nn.BatchNorm2d(planes)
        self.layerNr += 1

        self.conv2 = QuantizedConv2d(self.conv2_x, self.conv2_y, affected_rts=self.affected_rts, layerNr=self.layerNr, protectLayers=protectLayers, kernel_size=3,
                               stride=1, padding=1, quantization=quantMethod, an_sim=an_sim, array_size=array_size, mac_mapping=mapping, mac_mapping_distr=mapping_distr, sorted_mac_mapping_idx=sorted_mapping_idx,
                               performance_mode=performance_mode, test_rtm = test_rtm, index_offset = self.index_offset_conv2, rt_size = self.rt_size,
                               error_model=error_model, bias=False, train_model=train_model, extract_absfreq=extract_absfreq)
        self.bn2 = nn.BatchNorm2d(planes)
        self.layerNr += 1

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut_x = in_planes
            self.shortcut_y = self.expansion*planes
            self.resetShortcutOffsets()
            # print("--SHORTCUT--")
            # print(self.layerNr)
            self.shortcut = nn.Sequential(
                QuantizedConv2d(self.shortcut_x, self.shortcut_y, affected_rts=self.affected_rts, layerNr=self.layerNr, protectLayers=protectLayers,
                          kernel_size=1, stride=stride, quantization=quantMethod, an_sim=an_sim, array_size=array_size, mac_mapping=mapping, mac_mapping_distr=mapping_distr, sorted_mac_mapping_idx=sorted_mapping_idx,
                          performance_mode=performance_mode,
                          error_model=error_model, test_rtm = test_rtm, index_offset = self.index_offset_shortcut, rt_size = self.rt_size, bias=False, train_model=train_model, extract_absfreq=extract_absfreq),
                nn.BatchNorm2d(self.expansion*planes)
            )
            self.layerNr += 1


    def getRacetrackSize(self):
        return self.rt_size
    

    def getLayerNr(self):
        return self.layerNr

    def resetConvOffsets(self, in_planes, planes):
        self.conv1_x = in_planes
        self.conv1_y = planes
        self.conv2_x = planes
        self.conv2_y = planes

        self.index_offset_conv1 = np.zeros((math.ceil(self.conv1_x*self.conv1_y/self.rt_size), self.kernel_size * self.kernel_size))
        self.index_offset_conv2 = np.zeros((math.ceil(self.conv2_x*self.conv2_y/self.rt_size), self.kernel_size * self.kernel_size))


    def resetShortcutOffsets(self):
        if self.rt_size > 1*1*self.shortcut_x: # kernel size: 1x1
            shortcut_y = 1
        else:
            shortcut_y = int(1*1*self.shortcut_x/self.rt_size)
        self.index_offset_shortcut = np.zeros((self.shortcut_y, shortcut_y))
        # self.index_offset_shortcut = np.zeros((math.ceil(self.shortcut_x*self.shortcut_y/self.rt_size), 1))


    def forward(self, x):
        out = self.qact(self.htanh(self.bn1(self.conv1(x))))
        # out = self.qact(self.htanh(self.bn2(self.conv2(out))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.qact(self.htanh(out))
        return out
    

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, train_crit, test_crit, quantMethod=None,  quantize_train=True, quantize_eval=True, error_model=None, an_sim=None, array_size=None, mapping=None, mapping_distr=None, sorted_mapping_idx=None, performance_mode=None, train_model=None, extract_absfreq=None, num_classes=10, test_rtm = None, kernel_size=3, rt_size=64, protectLayers=[], affected_rts=[], misalign_faults=[], bitflips=[], global_bitflip_budget=0.05, local_bitflip_budget=0.1, calc_results=True, calc_bitflips=True, calc_misalign_faults=True, calc_affected_rts=True):
        super(ResNet, self).__init__()
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
        self.rt_size = rt_size
        self.kernel_size = kernel_size
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

        self.resetOffsets()

        self.htanh = nn.Hardtanh()
        self.qact = QuantizedActivation(quantization=self.quantization)

        self.layerNr = 1
        
        self.conv1 = QuantizedConv2d(self.conv1_x, self.conv1_y, affected_rts=self.affected_rts, layerNr=self.layerNr, protectLayers = self.protectLayers, kernel_size=self.kernel_size, stride=1, padding=1, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv1, rt_size = self.rt_size, bias=False, array_size=self.array_size, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts)
        self.bn1 = nn.BatchNorm2d(64)
        self.layerNr += 1
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, kernel_size=self.kernel_size, test_rtm=test_rtm)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, kernel_size=self.kernel_size, test_rtm=test_rtm)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, kernel_size=self.kernel_size, test_rtm=test_rtm)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, kernel_size=self.kernel_size, test_rtm=test_rtm)
        
        self.linear_x = 512*block.expansion
        self.linear_y = num_classes
        self.resetLinearOffsets()

        self.linear = QuantizedLinear(self.linear_x, self.linear_y, affected_rts=self.affected_rts, layerNr=self.layerNr, protectLayers=self.protectLayers, quantization=self.quantization, an_sim=self.an_sim, array_size=self.array_size, mac_mapping=self.mapping, mac_mapping_distr=self.mapping_distr, sorted_mac_mapping_idx=self.sorted_mapping_idx, performance_mode=self.performance_mode, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_linear, rt_size = self.rt_size, bias=False, train_model=self.train_model, extract_absfreq=self.extract_absfreq, misalign_faults=self.misalign_faults, bitflips=self.bitflips, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget, calc_results=calc_results, calc_bitflips=calc_bitflips, calc_misalign_faults=calc_misalign_faults, calc_affected_rts=calc_affected_rts)


    def _make_layer(self, block, planes, num_blocks, stride, kernel_size, test_rtm):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        # print(strides)
        for stride in strides:
            # print(stride)
            bblock = block(self.in_planes, planes, stride, kernel_size=kernel_size, quantMethod=self.quantization, an_sim=self.an_sim, array_size=self.array_size, mapping=self.mapping, mapping_distr=self.mapping_distr, sorted_mapping_idx=self.sorted_mapping_idx, performance_mode=self.performance_mode, error_model=self.error_model, train_model=self.train_model, extract_absfreq=self.extract_absfreq, test_rtm = test_rtm, rt_size=self.rt_size, layerNr = self.layerNr, protectLayers=self.protectLayers, affected_rts=self.affected_rts)
            layers.append(bblock)
            self.in_planes = planes * block.expansion
            self.layerNr = bblock.getLayerNr()
        return nn.Sequential(*layers)
    
    
    def getRacetrackSize(self):
        return self.rt_size
    

    def resetOffsets(self):
        self.conv1_x = 3
        self.conv1_y = 64
        self.index_offset_conv1 = np.zeros((math.ceil(self.conv1_x*self.conv1_y/self.rt_size), self.kernel_size * self.kernel_size))


    def resetLinearOffsets(self):
        self.index_offset_linear = np.zeros((self.linear_y, int(self.linear_x/self.rt_size)))
        # self.index_offset_linear = np.zeros((math.ceil(self.linear_x*self.linear_y/self.rt_size), 1))


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

