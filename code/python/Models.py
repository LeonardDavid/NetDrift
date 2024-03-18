import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation

class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class FC(nn.Module):
    def __init__(self, train_crit, test_crit, quantMethod=None, an_sim=None, array_size=None, mapping=None, quantize_train=True, quantize_eval=True, error_model=None):
        super(FC, self).__init__()
        self.name = "FC"
        self.traincriterion = train_crit
        self.testcriterion = test_crit
        self.quantization = quantMethod
        self.q_train = quantize_train
        self.q_test = quantize_eval
        self.error_model = error_model
        self.an_sim = an_sim
        self.array_size = array_size
        self.mapping = mapping
        self.htanh = nn.Hardtanh()

        self.flatten = torch.flatten
        self.fcfc1 = QuantizedLinear(28*28, 2048, quantization=self.quantization, an_sim=None, array_size=self.array_size, error_model=self.error_model, layerNr=1, bias=False)
        self.fcbn1 = nn.BatchNorm1d(2048)
        self.fcqact1 = QuantizedActivation(quantization=self.quantization)

        self.fcfc2 = QuantizedLinear(2048, 2048, quantization=self.quantization, an_sim=self.an_sim, array_size=self.array_size, mac_mapping=self.mapping, error_model=self.error_model, layerNr=2, bias=False)
        self.fcbn2 = nn.BatchNorm1d(2048)
        self.fcqact2 = QuantizedActivation(quantization=self.quantization)
        self.fcfc3 = QuantizedLinear(2048, 10, quantization=self.quantization, an_sim=self.an_sim, array_size=self.array_size, mac_mapping=self.mapping, layerNr=3, bias=False)
        self.scale = Scale()

    def forward(self, x):
        x = self.flatten(x, start_dim=1, end_dim=3)
        x = self.fcfc1(x)
        x = self.fcbn1(x)
        x = self.htanh(x)
        x = self.fcqact1(x)

        x = self.fcfc2(x)
        x = self.fcbn2(x)
        x = self.htanh(x)
        x = self.fcqact2(x)

        x = self.fcfc3(x)
        x = self.scale(x)

        return x

class VGG3(nn.Module):
    def __init__(self, train_crit, test_crit, quantMethod=None, an_sim=None, array_size=None, mapping=None, mapping_distr=None, sorted_mapping_idx=None, performance_mode=None, quantize_train=True, quantize_eval=True, error_model=None, train_model=None, extract_absfreq=None,  test_rtm = None, block_size=64, protectLayers=[]):
        super(VGG3, self).__init__()
        self.name = "VGG3"
        self.traincriterion = train_crit
        self.testcriterion = test_crit
        self.quantization = quantMethod
        self.q_train = quantize_train
        self.q_test = quantize_eval
        self.error_model = error_model
        self.an_sim = an_sim
        self.array_size = array_size
        self.mapping = mapping
        self.mapping_distr = mapping_distr
        self.sorted_mapping_idx = sorted_mapping_idx
        self.performance_mode = performance_mode
        self.train_model = train_model
        self.extract_absfreq = extract_absfreq
        self.htanh = nn.Hardtanh()
        self.block_size = block_size # 64
        self.resetOffsets()
        self.protectLayers = protectLayers

        self.conv1 = QuantizedConv2d(1, 64, layerNr=1, protectLayers = self.protectLayers, kernel_size=3, padding=1, stride=1, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv1, lost_vals_r = self.lost_vals_r_conv1, lost_vals_l = self.lost_vals_l_conv1, block_size = self.block_size, bias=False, array_size=self.array_size)
        self.bn1 = nn.BatchNorm2d(64)
        self.qact1 = QuantizedActivation(quantization=self.quantization)

        self.conv2 = QuantizedConv2d(64, 64, layerNr=2, protectLayers = self.protectLayers, kernel_size=3, padding=1, stride=1, quantization=self.quantization, an_sim=self.an_sim, array_size=self.array_size, mac_mapping=self.mapping, mac_mapping_distr=self.mapping_distr, sorted_mac_mapping_idx=self.sorted_mapping_idx, performance_mode=self.performance_mode, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv2, lost_vals_r = self.lost_vals_r_conv2, lost_vals_l = self.lost_vals_l_conv2, block_size = self.block_size, bias=False, train_model=self.train_model, extract_absfreq=self.extract_absfreq)
        self.bn2 = nn.BatchNorm2d(64)
        self.qact2 = QuantizedActivation(quantization=self.quantization)

        self.fc1 = QuantizedLinear(7*7*64, 2048, layerNr=3, protectLayers = self.protectLayers, quantization=self.quantization, an_sim=self.an_sim, array_size=self.array_size, mac_mapping=self.mapping, mac_mapping_distr=self.mapping_distr, sorted_mac_mapping_idx=self.sorted_mapping_idx, performance_mode=self.performance_mode, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_fc1, lost_vals_r = self.lost_vals_r_fc1, lost_vals_l = self.lost_vals_l_fc1, block_size = self.block_size, bias=False, train_model=self.train_model, extract_absfreq=self.extract_absfreq)
        self.bn3 = nn.BatchNorm1d(2048)
        self.qact3 = QuantizedActivation(quantization=self.quantization)

        self.fc2 = QuantizedLinear(2048, 10, layerNr=4, protectLayers = self.protectLayers, quantization=self.quantization, an_sim=self.an_sim, array_size=self.array_size, mac_mapping=self.mapping, mac_mapping_distr=self.mapping_distr, performance_mode=self.performance_mode, sorted_mac_mapping_idx=self.sorted_mapping_idx, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_fc2, lost_vals_r = self.lost_vals_r_fc2, lost_vals_l = self.lost_vals_l_fc2, block_size = self.block_size, bias=False, train_model=self.train_model, extract_absfreq=self.extract_absfreq)
        self.scale = Scale()

    
    def getBlockSize(self):
        return self.block_size
    
    def resetOffsets(self):
        # if self.conv1_size(0) >= 64:
        #     nr_blocks_conv1 = int(self.conv1_size(0)/self.block_size)
        # else:
        #     nr_blocks_conv1 = self.conv1_size
        # for conv 1 nr_blocks_conv1 has to be 1, because else it will set it to 0
        self.index_offset_conv1 = np.zeros((64, 1))
        self.lost_vals_r_conv1 = np.zeros((self.index_offset_conv1.shape[0], self.index_offset_conv1.shape[1]))
        self.lost_vals_l_conv1 = np.zeros((self.index_offset_conv1.shape[0], self.index_offset_conv1.shape[1]))

        self.index_offset_conv2 = np.zeros((64, int(64/self.block_size)))
        self.lost_vals_r_conv2 = np.zeros((self.index_offset_conv2.shape[0], self.index_offset_conv2.shape[1]))
        self.lost_vals_l_conv2 = np.zeros((self.index_offset_conv2.shape[0], self.index_offset_conv2.shape[1]))

        self.index_offset_fc1 = np.zeros((2048, int(7*7*64/self.block_size)))
        self.lost_vals_r_fc1 = np.zeros((self.index_offset_fc1.shape[0], self.index_offset_fc1.shape[1]))
        self.lost_vals_l_fc1 = np.zeros((self.index_offset_fc1.shape[0], self.index_offset_fc1.shape[1]))

        self.index_offset_fc2 = np.zeros((10, int(2048/self.block_size)))
        self.lost_vals_r_fc2 = np.zeros((self.index_offset_fc2.shape[0], self.index_offset_fc2.shape[1]))
        self.lost_vals_l_fc2 = np.zeros((self.index_offset_fc2.shape[0], self.index_offset_fc2.shape[1]))

    
    def getLostValsSum(self):
        return np.sum(self.lost_vals_l_conv1) + np.sum(self.lost_vals_l_conv2) + np.sum(self.lost_vals_l_fc1) + np.sum(self.lost_vals_l_fc2) + np.sum(self.lost_vals_r_fc1) + np.sum(self.lost_vals_r_fc2)
    
    def printIndexOffsets(self):
        print("conv1 " + str(self.index_offset_conv1.shape[0]) + " " + str(self.index_offset_conv1.shape[1]) + " " + str(np.sum(self.index_offset_conv1)))
        print(self.index_offset_conv1)
        print("conv2 " + str(self.index_offset_conv2.shape[0]) + " " + str(self.index_offset_conv2.shape[1]) + " " + str(np.sum(self.index_offset_conv2)))
        print(self.index_offset_conv2)
        # print("fc1 " + str(self.index_offset_fc1.shape[0]) + " " + str(self.index_offset_fc1.shape[1]) + " " + str(np.sum(self.index_offset_fc1)))
        # print(self.index_offset_fc1)
        # print("fc2 " + str(self.index_offset_fc2.shape[0]) + " " + str(self.index_offset_fc2.shape[1]) + " " + str(np.sum(self.index_offset_fc2)))
        # print(self.index_offset_fc2)

    def printLostValsR(self):
        print("lvr_conv1 " + str(self.lost_vals_r_conv1.shape[0]) + " " + str(self.lost_vals_r_conv1.shape[1]) + " " + str(np.sum(self.lost_vals_r_conv1)))
        print(self.lost_vals_r_conv1)
        print("lvr_conv2 " + str(self.lost_vals_r_conv2.shape[0]) + " " + str(self.lost_vals_r_conv2.shape[1]) + " " + str(np.sum(self.lost_vals_r_conv2)))
        print(self.lost_vals_r_conv2)
        # print("lvr_fc1 " + str(self.lost_vals_r_fc1.shape[0]) + " " + str(self.lost_vals_r_fc1.shape[1]) + " " + str(np.sum(self.lost_vals_r_fc1)))
        # print(self.lost_vals_r_fc1)
        # print("lvr_fc2 " + str(self.lost_vals_r_fc2.shape[0]) + " " + str(self.lost_vals_r_fc2.shape[1]) + " " + str(np.sum(self.lost_vals_r_fc2)))
        # print(self.lost_vals_r_fc2)

    def printLostValsL(self):
        print("lvl_conv1 " + str(self.lost_vals_l_conv1.shape[0]) + " " + str(self.lost_vals_l_conv1.shape[1]) + " " + str(np.sum(self.lost_vals_l_conv1)))
        print(self.lost_vals_l_conv1)
        print("lvl_conv2 " + str(self.lost_vals_l_conv2.shape[0]) + " " + str(self.lost_vals_l_conv2.shape[1]) + " " + str(np.sum(self.lost_vals_l_conv2)))
        print(self.lost_vals_l_conv2)
        # print("lvl_fc1 " + str(self.lost_vals_l_fc1.shape[0]) + " " + str(self.lost_vals_l_fc1.shape[1]) + " " + str(np.sum(self.lost_vals_l_fc1)))
        # print(self.lost_vals_l_fc1)
        # print("lvl_fc2 " + str(self.lost_vals_l_fc2.shape[0]) + " " + str(self.lost_vals_l_fc2.shape[1]) + " " + str(np.sum(self.lost_vals_l_fc2)))
        # print(self.lost_vals_l_fc2)

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
    def __init__(self, train_crit, test_crit, quantMethod=None, an_sim=None, array_size=None, mapping=None, mapping_distr=None, sorted_mapping_idx=None, performance_mode=None, quantize_train=True, quantize_eval=True, error_model=None, train_model=None, extract_absfreq=None, test_rtm = None, block_size=64, protectLayers=[]):
        super(VGG7, self).__init__()
        self.name = "VGG7"
        self.traincriterion = train_crit
        self.testcriterion = test_crit
        self.quantization = quantMethod
        self.q_train = quantize_train
        self.q_test = quantize_eval
        self.error_model = error_model
        self.an_sim = an_sim
        self.array_size = array_size
        self.mapping = mapping
        self.mapping_distr = mapping_distr
        self.sorted_mapping_idx = sorted_mapping_idx
        self.performance_mode = performance_mode
        self.train_model = train_model
        self.extract_absfreq = extract_absfreq
        self.htanh = nn.Hardtanh()
        self.block_size = block_size
        self.resetOffsets()
        self.protectLayers = protectLayers

        #CNN
        # block 1
        self.conv1 = QuantizedConv2d(3, 128, layerNr=1, protectLayers = self.protectLayers, kernel_size=3, padding=1, stride=1, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv1, lost_vals_r = self.lost_vals_r_conv1, lost_vals_l = self.lost_vals_l_conv1, block_size = self.block_size, bias=False, array_size=self.array_size)
        self.bn1 = nn.BatchNorm2d(128)
        self.qact1 = QuantizedActivation(quantization=self.quantization)

        # block 2
        self.conv2 = QuantizedConv2d(128, 128, layerNr=2, protectLayers = self.protectLayers, kernel_size=3, padding=1, stride=1, quantization=self.quantization, an_sim=self.an_sim, array_size=self.array_size, mac_mapping=self.mapping, mac_mapping_distr=self.mapping_distr, sorted_mac_mapping_idx=self.sorted_mapping_idx,
        performance_mode=self.performance_mode,
        error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv2, lost_vals_r = self.lost_vals_r_conv2, lost_vals_l = self.lost_vals_l_conv2, block_size = self.block_size, bias=False, train_model=self.train_model, extract_absfreq=self.extract_absfreq)
        self.bn2 = nn.BatchNorm2d(128)
        self.qact2 = QuantizedActivation(quantization=self.quantization)

        # block 3
        self.conv3 = QuantizedConv2d(128, 256, layerNr=3, protectLayers = self.protectLayers, kernel_size=3, padding=1, stride=1, quantization=self.quantization, an_sim=self.an_sim, array_size=self.array_size, mac_mapping=self.mapping, mac_mapping_distr=self.mapping_distr, sorted_mac_mapping_idx=self.sorted_mapping_idx,
        performance_mode=self.performance_mode, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv3, lost_vals_r = self.lost_vals_r_conv3, lost_vals_l = self.lost_vals_l_conv3, block_size = self.block_size, bias=False, train_model=self.train_model, extract_absfreq=self.extract_absfreq)
        self.bn3 = nn.BatchNorm2d(256)
        self.qact3 = QuantizedActivation(quantization=self.quantization)

        # block 4
        self.conv4 = QuantizedConv2d(256, 256, layerNr=4, protectLayers = self.protectLayers, kernel_size=3, padding=1, stride=1, quantization=self.quantization, an_sim=self.an_sim, array_size=self.array_size, mac_mapping=self.mapping, mac_mapping_distr=self.mapping_distr, sorted_mac_mapping_idx=self.sorted_mapping_idx,
        performance_mode=self.performance_mode, error_model=self.error_model,test_rtm = test_rtm, index_offset = self.index_offset_conv4, lost_vals_r = self.lost_vals_r_conv4, lost_vals_l = self.lost_vals_l_conv4, block_size = self.block_size, bias=False, train_model=self.train_model, extract_absfreq=self.extract_absfreq)
        self.bn4 = nn.BatchNorm2d(256)
        self.qact4 = QuantizedActivation(quantization=self.quantization)

        # block 5
        self.conv5 = QuantizedConv2d(256, 512, layerNr=5, protectLayers = self.protectLayers, kernel_size=3, padding=1, stride=1, quantization=self.quantization, an_sim=self.an_sim, array_size=self.array_size, mac_mapping=self.mapping, mac_mapping_distr=self.mapping_distr, sorted_mac_mapping_idx=self.sorted_mapping_idx,
        performance_mode=self.performance_mode, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv5, lost_vals_r = self.lost_vals_r_conv5, lost_vals_l = self.lost_vals_l_conv5, block_size = self.block_size, bias=False, train_model=self.train_model, extract_absfreq=self.extract_absfreq)
        self.bn5 = nn.BatchNorm2d(512)
        self.qact5 = QuantizedActivation(quantization=self.quantization)

        # block 6
        self.conv6 = QuantizedConv2d(512, 512, layerNr=6, protectLayers = self.protectLayers, kernel_size=3, padding=1, stride=1, quantization=self.quantization, an_sim=self.an_sim, array_size=self.array_size, mac_mapping=self.mapping, mac_mapping_distr=self.mapping_distr, sorted_mac_mapping_idx=self.sorted_mapping_idx,
        performance_mode=self.performance_mode, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv6, lost_vals_r = self.lost_vals_r_conv6, lost_vals_l = self.lost_vals_l_conv6, block_size = self.block_size, bias=False, train_model=self.train_model, extract_absfreq=self.extract_absfreq)
        self.bn6 = nn.BatchNorm2d(512)
        self.qact6 = QuantizedActivation(quantization=self.quantization)

        # block 7
        self.fc1 = QuantizedLinear(8192, 1024, layerNr=7, protectLayers = self.protectLayers, quantization=self.quantization, an_sim=self.an_sim, array_size=self.array_size, mac_mapping=self.mapping, mac_mapping_distr=self.mapping_distr, sorted_mac_mapping_idx=self.sorted_mapping_idx,
        performance_mode=self.performance_mode, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_fc1, lost_vals_r = self.lost_vals_r_fc1, lost_vals_l = self.lost_vals_l_fc1, block_size = self.block_size, bias=False, train_model=self.train_model, extract_absfreq=self.extract_absfreq)
        self.bn7 = nn.BatchNorm1d(1024)
        self.qact7 = QuantizedActivation(quantization=self.quantization)

        self.fc2 = QuantizedLinear(1024, 10, layerNr=8, protectLayers = self.protectLayers, quantization=self.quantization, an_sim=self.an_sim, array_size=self.array_size, mac_mapping=self.mapping, mac_mapping_distr=self.mapping_distr, sorted_mac_mapping_idx=self.sorted_mapping_idx,
        performance_mode=self.performance_mode, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_fc2, lost_vals_r = self.lost_vals_r_fc2, lost_vals_l = self.lost_vals_l_fc2, block_size = self.block_size, bias=False, train_model=self.train_model, extract_absfreq=self.extract_absfreq)
        self.scale = Scale(init_value=1e-3)


    def getBlockSize(self):
        return self.block_size
    
    
    def resetOffsets(self):
        # if self.conv1_size(0) >= 64:
        #     nr_blocks_conv1 = int(self.conv1_size(0)/self.block_size)
        # else:
        #     nr_blocks_conv1 = self.conv1_size
        # for conv 1 nr_blocks_conv1 has to be 3, because else it will set it to 0
        self.index_offset_conv1 = np.zeros((128, 3))
        self.lost_vals_r_conv1 = np.zeros((self.index_offset_conv1.shape[0], self.index_offset_conv1.shape[1]))
        self.lost_vals_l_conv1 = np.zeros((self.index_offset_conv1.shape[0], self.index_offset_conv1.shape[1]))

        self.index_offset_conv2 = np.zeros((128, int(128/self.block_size)))
        self.lost_vals_r_conv2 = np.zeros((self.index_offset_conv2.shape[0], self.index_offset_conv2.shape[1]))
        self.lost_vals_l_conv2 = np.zeros((self.index_offset_conv2.shape[0], self.index_offset_conv2.shape[1]))

        self.index_offset_conv3 = np.zeros((256, int(128/self.block_size)))
        self.lost_vals_r_conv3 = np.zeros((self.index_offset_conv3.shape[0], self.index_offset_conv3.shape[1]))
        self.lost_vals_l_conv3 = np.zeros((self.index_offset_conv3.shape[0], self.index_offset_conv3.shape[1]))

        self.index_offset_conv4 = np.zeros((256, int(256/self.block_size)))
        self.lost_vals_r_conv4 = np.zeros((self.index_offset_conv4.shape[0], self.index_offset_conv4.shape[1]))
        self.lost_vals_l_conv4 = np.zeros((self.index_offset_conv4.shape[0], self.index_offset_conv4.shape[1]))

        self.index_offset_conv5 = np.zeros((512, int(256/self.block_size)))
        self.lost_vals_r_conv5 = np.zeros((self.index_offset_conv5.shape[0], self.index_offset_conv5.shape[1]))
        self.lost_vals_l_conv5 = np.zeros((self.index_offset_conv5.shape[0], self.index_offset_conv5.shape[1]))

        self.index_offset_conv6 = np.zeros((512, int(512/self.block_size)))
        self.lost_vals_r_conv6 = np.zeros((self.index_offset_conv6.shape[0], self.index_offset_conv6.shape[1]))
        self.lost_vals_l_conv6 = np.zeros((self.index_offset_conv6.shape[0], self.index_offset_conv6.shape[1]))

        self.index_offset_fc1 = np.zeros((1024, int(8192/self.block_size)))
        self.lost_vals_r_fc1 = np.zeros((self.index_offset_fc1.shape[0], self.index_offset_fc1.shape[1]))
        self.lost_vals_l_fc1 = np.zeros((self.index_offset_fc1.shape[0], self.index_offset_fc1.shape[1]))

        self.index_offset_fc2 = np.zeros((10, int(1024/self.block_size)))
        self.lost_vals_r_fc2 = np.zeros((self.index_offset_fc2.shape[0], self.index_offset_fc2.shape[1]))
        self.lost_vals_l_fc2 = np.zeros((self.index_offset_fc2.shape[0], self.index_offset_fc2.shape[1]))


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

    def __init__(self, in_planes, planes, stride=1, quantMethod=None, an_sim=None, array_size=None, mapping=None, mapping_distr=None, sorted_mapping_idx=None, performance_mode=None, quantize_train=True, quantize_eval=True, error_model=None, train_model=None, extract_absfreq=None, test_rtm = None, block_size=64, layerNr=2, protectLayers=[], err_shifts=[]):
        super(BasicBlock, self).__init__()
        self.htanh = nn.Hardtanh()
        self.block_size = block_size #64
        self.layerNr = layerNr
        self.protectLayers = protectLayers
        self.err_shifts = err_shifts
        # print("##BLOCK##")
    
        self.qact = QuantizedActivation(quantization=quantMethod)

        self.conv1_size_1 = in_planes
        self.conv1_size_2 = planes
        self.resetConv1Offsets()
        # print(self.layerNr)
        self.conv1 = QuantizedConv2d(
            self.conv1_size_1, self.conv1_size_2, err_shifts=self.err_shifts, layerNr=self.layerNr, protectLayers=protectLayers, kernel_size=3, stride=stride, padding=1, quantization=quantMethod, an_sim=an_sim, array_size=array_size, mac_mapping=mapping, mac_mapping_distr=mapping_distr, sorted_mac_mapping_idx=sorted_mapping_idx,
            performance_mode=performance_mode, test_rtm = test_rtm, index_offset = self.index_offset_conv1, lost_vals_r = self.lost_vals_r_conv1, lost_vals_l = self.lost_vals_l_conv1, block_size = self.block_size,
            error_model=error_model, bias=False, train_model=train_model, extract_absfreq=extract_absfreq)
        self.bn1 = nn.BatchNorm2d(planes)
        self.layerNr += 1

        self.conv2_size_1 = planes
        self.conv2_size_2 = planes
        self.resetConv2Offsets()
        # print(self.layerNr)
        self.conv2 = QuantizedConv2d(self.conv2_size_1, self.conv2_size_2, err_shifts=self.err_shifts, layerNr=self.layerNr, protectLayers=protectLayers, kernel_size=3,
                               stride=1, padding=1, quantization=quantMethod, an_sim=an_sim, array_size=array_size, mac_mapping=mapping, mac_mapping_distr=mapping_distr, sorted_mac_mapping_idx=sorted_mapping_idx,
                               performance_mode=performance_mode, test_rtm = test_rtm, index_offset = self.index_offset_conv2, lost_vals_r = self.lost_vals_r_conv2, lost_vals_l = self.lost_vals_l_conv2, block_size = self.block_size,
                               error_model=error_model, bias=False, train_model=train_model, extract_absfreq=extract_absfreq)
        self.bn2 = nn.BatchNorm2d(planes)
        self.layerNr += 1

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut_size_1 = in_planes
            self.shortcut_size_2 = self.expansion*planes
            self.resetShortcutOffsets()
            # print("--SHORTCUT--")
            # print(self.layerNr)
            self.shortcut = nn.Sequential(
                QuantizedConv2d(self.shortcut_size_1, self.shortcut_size_2, err_shifts=self.err_shifts, layerNr=self.layerNr, protectLayers=protectLayers,
                          kernel_size=1, stride=stride, quantization=quantMethod, an_sim=an_sim, array_size=array_size, mac_mapping=mapping, mac_mapping_distr=mapping_distr, sorted_mac_mapping_idx=sorted_mapping_idx,
                          performance_mode=performance_mode,
                          error_model=error_model, test_rtm = test_rtm, index_offset = self.index_offset_shortcut, lost_vals_r = self.lost_vals_r_shortcut, lost_vals_l = self.lost_vals_l_shortcut, block_size = self.block_size, bias=False, train_model=train_model, extract_absfreq=extract_absfreq),
                nn.BatchNorm2d(self.expansion*planes)
            )
            self.layerNr += 1

    def getBlockSize(self):
        return self.block_size
    
    def getLayerNr(self):
        return self.layerNr
    
    def resetConv1Offsets(self):
        # if self.conv1_size(0) >= 64:
        #     nr_blocks_conv1 = int(self.conv1_size(0)/self.block_size)
        # else:
        #     nr_blocks_conv1 = self.conv1_size
        # for conv 1 nr_blocks_conv1 has to be 1, because else it will set it to 0
        self.index_offset_conv1 = np.zeros((self.conv1_size_2, self.conv1_size_1))
        self.lost_vals_r_conv1 = np.zeros((self.index_offset_conv1.shape[0], self.index_offset_conv1.shape[1]))
        self.lost_vals_l_conv1 = np.zeros((self.index_offset_conv1.shape[0], self.index_offset_conv1.shape[1]))

    def resetConv2Offsets(self): 
        
        if self.block_size > self.conv2_size_1:
            conv2_y = 1
        else:
            conv2_y = int(self.conv2_size_1/self.block_size)
        self.index_offset_conv2 = np.zeros((self.conv2_size_2, conv2_y))
        self.lost_vals_r_conv2 = np.zeros((self.index_offset_conv2.shape[0], self.index_offset_conv2.shape[1]))
        self.lost_vals_l_conv2 = np.zeros((self.index_offset_conv2.shape[0], self.index_offset_conv2.shape[1]))

    def resetShortcutOffsets(self):

        if self.block_size > self.shortcut_size_1:
            shortcut_y = 1
        else:
            shortcut_y = int(self.shortcut_size_1/self.block_size)
        self.index_offset_shortcut = np.zeros((self.shortcut_size_2, shortcut_y))
        self.lost_vals_r_shortcut = np.zeros((self.index_offset_shortcut.shape[0], self.index_offset_shortcut.shape[1]))
        self.lost_vals_l_shortcut = np.zeros((self.index_offset_shortcut.shape[0], self.index_offset_shortcut.shape[1]))


    def forward(self, x):
        out = self.qact(self.htanh(self.bn1(self.conv1(x))))
        # out = self.qact(self.htanh(self.bn2(self.conv2(out))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.qact(self.htanh(out))
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, train_crit, test_crit, quantMethod=None, an_sim=None, array_size=None, mapping=None, mapping_distr=None, sorted_mapping_idx=None, performance_mode=None, quantize_train=True, quantize_eval=True, error_model=None, train_model=None, extract_absfreq=None, num_classes=10, test_rtm = None, block_size=64, protectLayers=[], err_shifts=[]):
        super(ResNet, self).__init__()
        self.name = "ResNet18"
        self.traincriterion = train_crit
        self.testcriterion = test_crit
        self.quantization = quantMethod
        self.q_train = quantize_train
        self.q_test = quantize_eval
        self.error_model = error_model
        self.an_sim = an_sim
        self.array_size = array_size
        self.mapping = mapping
        self.mapping_distr = mapping_distr
        self.sorted_mapping_idx = sorted_mapping_idx
        self.performance_mode = performance_mode
        self.train_model = train_model
        self.extract_absfreq = extract_absfreq
        self.in_planes = 64
        self.block_size = block_size #64
        self.resetOffsets()
        self.protectLayers = protectLayers
        self.err_shifts = err_shifts

        self.htanh = nn.Hardtanh()
        self.qact = QuantizedActivation(quantization=self.quantization)

        # print(self.error_model)

        self.layerNr = 1
        # print(self.layerNr)
        self.conv1 = QuantizedConv2d(3, 64, err_shifts=self.err_shifts, layerNr=self.layerNr, protectLayers = self.protectLayers, kernel_size=3, stride=1, padding=1, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv1, lost_vals_r = self.lost_vals_r_conv1, lost_vals_l = self.lost_vals_l_conv1, block_size = self.block_size, bias=False, array_size=self.array_size)
        self.bn1 = nn.BatchNorm2d(64)
        self.layerNr += 1
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, test_rtm=test_rtm)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, test_rtm=test_rtm)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, test_rtm=test_rtm)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, test_rtm=test_rtm)
        
        self.linear_size_1 = 512*block.expansion
        self.linear_size_2 = num_classes
        self.resetLinearOffsets()
        # print(self.layerNr)
        self.linear = QuantizedLinear(self.linear_size_1, self.linear_size_2, err_shifts=self.err_shifts, layerNr=self.layerNr, protectLayers=self.protectLayers, quantization=self.quantization, an_sim=self.an_sim, array_size=self.array_size, mac_mapping=self.mapping, mac_mapping_distr=self.mapping_distr, sorted_mac_mapping_idx=self.sorted_mapping_idx, performance_mode=self.performance_mode, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_linear, lost_vals_r = self.lost_vals_r_linear, lost_vals_l = self.lost_vals_l_linear, block_size = self.block_size, bias=False, train_model=self.train_model, extract_absfreq=self.extract_absfreq)

    def _make_layer(self, block, planes, num_blocks, stride, test_rtm):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        # print(strides)
        for stride in strides:
            # print(stride)
            bblock = block(self.in_planes, planes, stride, quantMethod=self.quantization, an_sim=self.an_sim, array_size=self.array_size, mapping=self.mapping, mapping_distr=self.mapping_distr, sorted_mapping_idx=self.sorted_mapping_idx, performance_mode=self.performance_mode, error_model=self.error_model, train_model=self.train_model, extract_absfreq=self.extract_absfreq, test_rtm = test_rtm, block_size=self.block_size, layerNr = self.layerNr, protectLayers=self.protectLayers, err_shifts=self.err_shifts)
            layers.append(bblock)
            self.in_planes = planes * block.expansion
            self.layerNr = bblock.getLayerNr()
        return nn.Sequential(*layers)
    
    
    def getBlockSize(self):
        return self.block_size
    
    
    def resetOffsets(self):
        # if self.conv1_size(0) >= 64:
        #     nr_blocks_conv1 = int(self.conv1_size(0)/self.block_size)
        # else:
        #     nr_blocks_conv1 = self.conv1_size
        # for conv 1 nr_blocks_conv1 has to be 3, because else it will set it to 0

        conv1_y = int(64/self.block_size)
        self.index_offset_conv1 = np.zeros((3, conv1_y))
        self.lost_vals_r_conv1 = np.zeros((self.index_offset_conv1.shape[0], self.index_offset_conv1.shape[1]))
        self.lost_vals_l_conv1 = np.zeros((self.index_offset_conv1.shape[0], self.index_offset_conv1.shape[1]))

    def resetLinearOffsets(self):
        self.index_offset_linear = np.zeros((self.linear_size_2, int(self.linear_size_1/self.block_size)))
        self.lost_vals_r_linear = np.zeros((self.index_offset_linear.shape[0], self.index_offset_linear.shape[1]))
        self.lost_vals_l_linear = np.zeros((self.index_offset_linear.shape[0], self.index_offset_linear.shape[1]))

    def printIndexOffsets(self):
        print("conv1 " + str(self.index_offset_conv1.shape[0]) + " " + str(self.index_offset_conv1.shape[1]) + " " + str(np.sum(self.index_offset_conv1)))
        print(self.index_offset_conv1)

    def printLostValsR(self):
        print("lvr_conv1 " + str(self.lost_vals_r_conv1.shape[0]) + " " + str(self.lost_vals_r_conv1.shape[1]) + " " + str(np.sum(self.lost_vals_r_conv1)))
        print(self.lost_vals_r_conv1)

    def printLostValsL(self):
        print("lvl_conv1 " + str(self.lost_vals_l_conv1.shape[0]) + " " + str(self.lost_vals_l_conv1.shape[1]) + " " + str(np.sum(self.lost_vals_l_conv1)))
        print(self.lost_vals_l_conv1)


    def forward(self, x):
        out = self.qact(self.htanh(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        # out = F.max_pool2d(out, 2)
        out = self.layer2(out)
        # out = F.max_pool2d(out, 2)
        out = self.layer3(out)
        # out = F.max_pool2d(out, 2)
        out = F.max_pool2d(out, 2)
        out = self.layer4(out)
        out = F.max_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # print("---")
        return out


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1, quantMethod=None, an_sim=None, array_size=None, mapping=None, mapping_distr=None, sorted_mapping_idx=None, performance_mode=None, quantize_train=True, quantize_eval=True, error_model=None, train_model=None, extract_absfreq=None, test_rtm = None, block_size=64):
#         super(BasicBlock, self).__init__()
#         self.htanh = nn.Hardtanh()
#         self.block_size = block_size #64
    
#         self.qact = QuantizedActivation(quantization=quantMethod)

#         self.conv1 = QuantizedConv2d(
#             self.conv1_size_1, self.conv1_size_2, kernel_size=3, stride=stride, padding=1, quantization=quantMethod, an_sim=an_sim, array_size=array_size, mac_mapping=mapping, mac_mapping_distr=mapping_distr, sorted_mac_mapping_idx=sorted_mapping_idx,
#             performance_mode=performance_mode, test_rtm = test_rtm, index_offset = self.index_offset_conv1, lost_vals_r = self.lost_vals_r_conv1, lost_vals_l = self.lost_vals_l_conv1, block_size = self.block_size,
#             error_model=error_model, bias=False, train_model=train_model, extract_absfreq=extract_absfreq)
#         self.bn1 = nn.BatchNorm2d(planes)
        
#         self.conv2_size_1 = planes
#         self.conv2_size_2 = planes
#         self.conv2 = QuantizedConv2d(self.conv2_size_1, self.conv2_size_2, kernel_size=3,
#                                stride=1, padding=1, quantization=quantMethod, an_sim=an_sim, array_size=array_size, mac_mapping=mapping, mac_mapping_distr=mapping_distr, sorted_mac_mapping_idx=sorted_mapping_idx,
#                                performance_mode=performance_mode, test_rtm = test_rtm, index_offset = self.index_offset_conv2, lost_vals_r = self.lost_vals_r_conv2, lost_vals_l = self.lost_vals_l_conv2, block_size = self.block_size,
#                                error_model=error_model, bias=False, train_model=train_model, extract_absfreq=extract_absfreq)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut_size_1 = in_planes
#             self.shortcut_size_2 = self.expansion*planes
#             self.shortcut = nn.Sequential(
#                 QuantizedConv2d(self.shortcut_size_1, self.shortcut_size_2,
#                           kernel_size=1, stride=stride, quantization=quantMethod, an_sim=an_sim, array_size=array_size, mac_mapping=mapping, mac_mapping_distr=mapping_distr, sorted_mac_mapping_idx=sorted_mapping_idx,
#                           performance_mode=performance_mode,
#                           error_model=error_model, test_rtm = test_rtm, index_offset = self.index_offset_shortcut, lost_vals_r = self.lost_vals_r_shortcut, lost_vals_l = self.lost_vals_l_shortcut, block_size = self.block_size, bias=False, train_model=train_model, extract_absfreq=extract_absfreq),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def getBlockSize(self):
#         return self.block_size
    



#     def forward(self, x):
#         out = self.qact(self.htanh(self.bn1(self.conv1(x))))
#         # out = self.qact(self.htanh(self.bn2(self.conv2(out))))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = self.qact(self.htanh(out))
#         return out

# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, train_crit, test_crit, quantMethod=None, an_sim=None, array_size=None, mapping=None, mapping_distr=None, sorted_mapping_idx=None, performance_mode=None, quantize_train=True, quantize_eval=True, error_model=None, train_model=None, extract_absfreq=None, num_classes=10, test_rtm = None, block_size=64):
#         super(ResNet, self).__init__()
#         self.name = "ResNet18"
#         self.traincriterion = train_crit
#         self.testcriterion = test_crit
#         self.quantization = quantMethod
#         self.q_train = quantize_train
#         self.q_test = quantize_eval
#         self.error_model = error_model
#         self.an_sim = an_sim
#         self.array_size = array_size
#         self.mapping = mapping
#         self.mapping_distr = mapping_distr
#         self.sorted_mapping_idx = sorted_mapping_idx
#         self.performance_mode = performance_mode
#         self.train_model = train_model
#         self.extract_absfreq = extract_absfreq
#         self.in_planes = 64
#         self.block_size = block_size #64
#         self.resetOffsets()

#         self.htanh = nn.Hardtanh()
#         self.qact = QuantizedActivation(quantization=self.quantization)

#         self.conv1 = QuantizedConv2d(3, 64, kernel_size=3, stride=1, padding=1, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv1, lost_vals_r = self.lost_vals_r_conv1, lost_vals_l = self.lost_vals_l_conv1, block_size = self.block_size, bias=False, array_size=self.array_size)
#         self.bn1 = nn.BatchNorm2d(64)
        
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, test_rtm=test_rtm)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, test_rtm=test_rtm)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, test_rtm=test_rtm)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, test_rtm=test_rtm)
        
#         self.linear_size_1 = 512*block.expansion
#         self.linear_size_2 = num_classes
#         self.linear = QuantizedLinear(self.linear_size_1, self.linear_size_2, quantization=self.quantization, an_sim=self.an_sim, array_size=self.array_size, mac_mapping=self.mapping, mac_mapping_distr=self.mapping_distr, sorted_mac_mapping_idx=self.sorted_mapping_idx, performance_mode=self.performance_mode, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_linear, lost_vals_r = self.lost_vals_r_linear, lost_vals_l = self.lost_vals_l_linear, block_size = self.block_size, bias=False, train_model=self.train_model, extract_absfreq=self.extract_absfreq)

#     def _make_layer(self, block, planes, num_blocks, stride, test_rtm):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride, quantMethod=self.quantization, an_sim=self.an_sim, array_size=self.array_size, mapping=self.mapping, mapping_distr=self.mapping_distr, sorted_mapping_idx=self.sorted_mapping_idx, performance_mode=self.performance_mode, error_model=self.error_model, train_model=self.train_model, extract_absfreq=self.extract_absfreq, test_rtm = test_rtm, block_size=self.block_size))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)
    
    
#     def getBlockSize(self):
#         return self.block_size
    
    
#     def resetOffsets(self):
#         # if self.conv1_size(0) >= 64:
#         #     nr_blocks_conv1 = int(self.conv1_size(0)/self.block_size)
#         # else:
#         #     nr_blocks_conv1 = self.conv1_size
#         # for conv 1 nr_blocks_conv1 has to be 3, because else it will set it to 0
#         self.index_offset_conv1 = np.zeros((64, 3))
#         self.lost_vals_r_conv1 = np.zeros((self.index_offset_conv1.shape[0], self.index_offset_conv1.shape[1]))
#         self.lost_vals_l_conv1 = np.zeros((self.index_offset_conv1.shape[0], self.index_offset_conv1.shape[1]))

#         # if self.conv1_size(0) >= 64:
#         #     nr_blocks_conv1 = int(self.conv1_size(0)/self.block_size)
#         # else:
#         #     nr_blocks_conv1 = self.conv1_size
#         # for conv 1 nr_blocks_conv1 has to be 1, because else it will set it to 0
#         self.index_offset_conv1_bb = np.zeros((self.conv1_size_2_bb, self.conv1_size_1_bb))
#         self.lost_vals_r_conv1_bb = np.zeros((self.index_offset_conv1_bb.shape[0], self.index_offset_conv1_bb.shape[1]))
#         self.lost_vals_l_conv1_bb = np.zeros((self.index_offset_conv1_bb.shape[0], self.index_offset_conv1_bb.shape[1]))

#         self.index_offset_conv2_bb = np.zeros((self.conv2_size_2_bb, int(self.conv2_size_1_bb/self.block_size)))
#         self.lost_vals_r_conv2_bb = np.zeros((self.index_offset_conv2_bb.shape[0], self.index_offset_conv2_bb.shape[1]))
#         self.lost_vals_l_conv2_bb = np.zeros((self.index_offset_conv2_bb.shape[0], self.index_offset_conv2_bb.shape[1]))

#         self.index_offset_shortcut_bb = np.zeros((self.shortcut_size_2_bb, int(self.shortcut_size_1_bb/self.block_size)))
#         self.lost_vals_r_shortcut_bb = np.zeros((self.index_offset_shortcut_bb.shape[0], self.index_offset_shortcut_bb.shape[1]))
#         self.lost_vals_l_shortcut_bb = np.zeros((self.index_offset_shortcut_bb.shape[0], self.index_offset_shortcut_bb.shape[1]))

#         self.index_offset_linear = np.zeros((self.linear_size_2, int(self.linear_size_1/self.block_size)))
#         self.lost_vals_r_linear = np.zeros((self.index_offset_linear.shape[0], self.index_offset_linear.shape[1]))
#         self.lost_vals_l_linear = np.zeros((self.index_offset_linear.shape[0], self.index_offset_linear.shape[1]))

#     def printIndexOffsets(self):
#         print("conv1 " + str(self.index_offset_conv1.shape[0]) + " " + str(self.index_offset_conv1.shape[1]) + " " + str(np.sum(self.index_offset_conv1)))
#         print(self.index_offset_conv1)

#     def printLostValsR(self):
#         print("lvr_conv1 " + str(self.lost_vals_r_conv1.shape[0]) + " " + str(self.lost_vals_r_conv1.shape[1]) + " " + str(np.sum(self.lost_vals_r_conv1)))
#         print(self.lost_vals_r_conv1)

#     def printLostValsL(self):
#         print("lvl_conv1 " + str(self.lost_vals_l_conv1.shape[0]) + " " + str(self.lost_vals_l_conv1.shape[1]) + " " + str(np.sum(self.lost_vals_l_conv1)))
#         print(self.lost_vals_l_conv1)


#     def forward(self, x):
#         out = self.qact(self.htanh(self.bn1(self.conv1(x))))
#         out = self.layer1(out)
#         # out = F.max_pool2d(out, 2)
#         out = self.layer2(out)
#         # out = F.max_pool2d(out, 2)
#         out = self.layer3(out)
#         # out = F.max_pool2d(out, 2)
#         out = F.max_pool2d(out, 2)
#         out = self.layer4(out)
#         out = F.max_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         # print("---")
#         return out
