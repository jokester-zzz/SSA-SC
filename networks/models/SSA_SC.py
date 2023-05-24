#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D
import torch_scatter
from networks.models.SC_2D_UNet import BEV_UNet
from networks.models.SC_3D_UNet import Asymm_3d_spconv
from networks.common.lovasz_losses import lovasz_softmax

class SegmentationHead(nn.Module):
    '''
  3D Segmentation heads to retrieve semantic segmentation at each scale.
  Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
  '''
    def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list):
        # 1， 8， 20， [1, 2, 3]
        super().__init__()

        # First convolution
        self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)

        # ASPP Block
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList([nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
        self.bn1 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
        self.conv2 = nn.ModuleList([nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
        self.bn2 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
        self.relu = nn.ReLU(inplace=True)

        # Convolution for output
        self.conv_classes = nn.Conv3d(planes, nbr_classes, kernel_size=3, padding=1, stride=1)

    def forward(self, x_in):

        # Dimension exapension
        x_in = x_in[:, None, :, :, :]  # (4, 1, 32, 256, 256)

        # Convolution to go from inplanes to planes features...
        x_in = self.relu(self.conv0(x_in))  # (4, 8, 32, 256, 256)

        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))  # (4, 8, 32, 256, 256)
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
        x_in = self.relu(y + x_in)  # modified

        x_in = self.conv_classes(x_in)  # (4, 20, 32, 256, 256)

        return x_in

class SSA_SC(nn.Module):
    def __init__(self, class_num, input_dimensions, class_frequencies, pt_model='pointnet', fea_dim=3, pt_pooling='max', kernal_size=3, out_pt_fea_dim=128, fea_compre=32):
        super().__init__()
        self.nbr_classes = class_num
        self.input_dimensions = np.array(input_dimensions)  # Grid dimensions should be (W, H, D).. z or height being axis 1   256*32*256
        self.class_frequencies = class_frequencies

        self.n_height = self.input_dimensions[1]
        self.dilation = 1
        self.bilinear = True
        self.group_conv = False 
        self.input_batch_norm = True
        self.dropout = 0.5
        self.circular_padding = False
        self.dropblock = False
        
        self.SSCNet = BEV_UNet(self.nbr_classes*self.n_height, self.n_height, self.dilation, self.bilinear, self.group_conv,
                            self.input_batch_norm, self.dropout, self.circular_padding, self.dropblock)

        self.SegNet = Asymm_3d_spconv(input_dimensions=self.input_dimensions, input_channels=fea_compre, num_classes=self.nbr_classes-1)

        if pt_model == 'pointnet':
            self.PPmodel = nn.Sequential(
                nn.BatchNorm1d(fea_dim),
                
                nn.Linear(fea_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                
                nn.Linear(256, out_pt_fea_dim)
            )

        self.pt_model = pt_model
        self.pt_pooling = pt_pooling
        self.fea_compre = fea_compre    # 32

        # NN stuff
        if kernal_size != 1:
            if self.pt_pooling == 'max':
                self.local_pool_op = torch.nn.MaxPool2d(kernal_size, stride=1, padding=(kernal_size-1)//2, dilation=1)
            else: raise NotImplementedError
        else: self.local_pool_op = None
        
        # parametric pooling        
        if self.pt_pooling == 'max':
            self.pool_dim = out_pt_fea_dim
        
        # point feature compression
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(nn.Linear(self.pool_dim, self.fea_compre), nn.ReLU())
            self.vox_fea_compression = nn.Sequential(nn.Linear(self.pool_dim, self.fea_compre), nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim
        
    def forward(self, x, stat='train'):

        cur_dev = x['3D_OCCUPANCY'].get_device()
        xy_ind = x['grid_ind']
        xyz_ind = x['voxel_ind']
        pt_fea = x['feature']
        occu = x['3D_OCCUPANCY'].squeeze(1).permute(0,2,1,3)    # [B, 1, 256, 32, 256]   -> [B, 32, 256, 256]
        
        # =====================Preperation for data===================
        # concate everything
        cat_pt_ind = []
        cat_pt_xyz_ind = []
        for i_batch in range(len(xy_ind)):
            cat_pt_ind.append(F.pad(xy_ind[i_batch], (1, 0), 'constant', value=i_batch))    # B,N,3 every point_ind is pad with its batch_id
            cat_pt_xyz_ind.append(F.pad(xyz_ind[i_batch], (1, 0), 'constant', value=i_batch))    # B,N,4 every point_ind is pad with its batch_id
    
        cat_pt_fea = torch.cat(pt_fea, dim=0)    # B*N, 7
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)    # B*N, 3
        cat_pt_xyz_ind = torch.cat(cat_pt_xyz_ind, dim = 0)    # B*N, 4

        pt_num = cat_pt_ind.shape[0]

        # shuffle the data
        shuffled_ind = torch.randperm(pt_num, device=cur_dev)
        cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :]
        cat_pt_xyz_ind = cat_pt_xyz_ind[shuffled_ind, :]
        
        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)

        unq_xyz, unq_inv_xyz, unq_cnt_xyz = torch.unique(cat_pt_xyz_ind, return_inverse=True, return_counts=True, dim=0)
        unq_xyz = unq_xyz.type(torch.int64)

        # process feature
        if self.pt_model == 'pointnet':
            processed_cat_pt_fea = self.PPmodel(cat_pt_fea)    # B*N,512
        
        if self.pt_pooling == 'max':
            pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]    # num_unq, 512
            voxel_features = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv_xyz, dim=0)[0]
        else: raise NotImplementedError
        
        if self.fea_compre:
            processed_pooled_data = self.fea_compression(pooled_data)    # num_unq, 32
            processed_voxel_features = self.vox_fea_compression(voxel_features)
        else:
            processed_pooled_data = pooled_data
        
        # stuff pooled data into 4D tensor
        out_data_dim = [len(pt_fea), self.input_dimensions[0], self.input_dimensions[2], self.pt_fea_dim]    # [B, 256, 256， 32]
        out_data = torch.zeros(out_data_dim, dtype=torch.float32).to(cur_dev)
        out_data[unq[:, 0], unq[:, 1], unq[:, 2], :] = processed_pooled_data    # [B, 256, 256， 32]
        out_data = out_data.permute(0, 3, 1, 2)    # [B, 32, 256, 256]

        if self.local_pool_op != None:
            out_data = self.local_pool_op(out_data)
    
        out_data = torch.cat((occu, out_data), 1)

        # ============================ For 3D Semantic Segmentation Network ==========================
        """
        Args:
            batch_dict:
                batch_size: intis
                voxel_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        """
        batch_dict = {}
        batch_dict['batch_size'] = len(pt_fea)
        batch_dict['voxel_features'] = processed_voxel_features  # num_voxels, 32
        batch_dict['voxel_coords'] = unq_xyz   # num_voxels, 4

        # ============================ Run Through Segmentation Network ============================

        out_seg_data, x_ds1, x_ds2, x_ds3 = self.SegNet(batch_dict, stat='test')    # B,19,256,256,32
        x_ds1 = x_ds1.dense().permute(0,1,4,2,3)    # [B, 32, 16, 128, 128]    
        x_ds2 = x_ds2.dense().permute(0,1,4,2,3)    # [B, 64, 8, 64, 64]  
        x_ds3 = x_ds3.dense().permute(0,1,4,2,3)    # [B, 128, 4, 32, 32]  
        # x_ds4 = x_ds4.dense()    
        x_ds1 = x_ds1.reshape(-1, x_ds1.shape[1]*x_ds1.shape[2], x_ds1.shape[3], x_ds1.shape[4])    # [B, 512, 128, 128]
        x_ds2 = x_ds2.reshape(-1, x_ds2.shape[1]*x_ds2.shape[2], x_ds2.shape[3], x_ds2.shape[4])    # [B, 512, 64, 64]
        x_ds3 = x_ds3.reshape(-1, x_ds3.shape[1]*x_ds3.shape[2], x_ds3.shape[3], x_ds3.shape[4])    # [B, 512, 32, 32]

        # ============================ Run Through Completion Network ============================
        x = self.SSCNet(out_data, x_ds1, x_ds2, x_ds3)   # [B, 640, 256, 256]
        x = x.permute(0,2,3,1)   # [B, 256, 256, 640]
        new_shape = list(x.shape)[:3] + [self.n_height, self.nbr_classes]    # [B, 256, 256, 32, 20]
        x = x.view(new_shape)    # [B, 256, 256, 32, 20]
        out_scale_1_1_3D = x.permute(0,4,1,3,2)   # [B, 20, 256, 32, 256]  # [B, C, H, W, D] -> [B, C, W, H, D]

        scores = [{'pred_semantic_1_1': out_scale_1_1_3D}, out_seg_data]

        return scores  # [B, 20, 256, 32, 256]

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        b, in_channels, h, w, d = x.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = x.view(b, out_channels, -1, h, w, d).sum(dim=2)
        return x

    def weights_initializer(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def weights_init(self):
        self.apply(self.weights_initializer)

    def get_parameters(self):
        return self.parameters()

    def compute_loss(self, scores, data):
        '''
        :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
        '''

        if len(scores) == 2:
            out_seg_data = scores[1]
            scores = scores[0]

        target = data['3D_LABEL']['1_1']
        device, dtype = target.device, target.dtype
        class_weights = self.get_class_weights().to(device=target.device, dtype=target.dtype)

        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, reduction='mean').to(device=device)
        loss_fun = nn.CrossEntropyLoss(ignore_index=255).to(device=device)

        loss_1_1 = criterion(scores['pred_semantic_1_1'], data['3D_LABEL']['1_1'].long())
        loss_1_1 += lovasz_softmax(torch.nn.functional.softmax(scores['pred_semantic_1_1'], dim=1), data['3D_LABEL']['1_1'].long(), ignore=255)

        loss_seg = loss_1_1
        loss_total = loss_1_1
        loss = {'total': loss_total, 'semantic_1_1': loss_1_1, 'semantic_seg': loss_seg}

        return loss

    def get_class_weights(self):
        '''
        Class weights being 1/log(fc) (https://arxiv.org/pdf/2008.10559.pdf)
        '''
        epsilon_w = 0.001  # eps to avoid zero division
        weights = torch.from_numpy(1 / np.log(self.class_frequencies + epsilon_w))

        return weights

    def get_target(self, data):
        '''
        Return the target to use for evaluation of the model
        '''
        return {'1_1': data['3D_LABEL']['1_1']}

    def get_scales(self):
        '''
        Return scales needed to train the model
        '''
        scales = ['1_1']
        return scales

    def get_validation_loss_keys(self):
        return ['total', 'semantic_1_1', 'semantic_seg']

    def get_train_loss_keys(self):
        return ['total', 'semantic_1_1', 'semantic_seg']

