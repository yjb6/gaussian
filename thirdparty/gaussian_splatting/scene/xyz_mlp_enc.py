#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH

from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from sklearn.neighbors import KDTree
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, update_quaternion
from helper_model import getcolormodel, interpolate_point, interpolate_partuse,interpolate_pointv3
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.xyz_activation = torch.sigmoid
        def xavier_init(m):
            for name,W in m.named_parameters():
                if 'bias' in name:
                    torch.nn.init.constant_(W, 0)
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(W)
        def uniform_init(m):
            for name,W in m.named_parameters():
                if 'bias' in name:
                    torch.nn.init.constant_(W, 0)
                if 'weight' in name:
                    torch.nn.init.uniform_(W)

        def constant_init(m,value):
            for name,W in m.named_parameters():
                torch.nn.init.constant_(W, value)
        self.xavier_init = xavier_init
        self.uniform_init = uniform_init
        self.constant_init = constant_init
        #self.featureact = torch.sigmoid

    def params_init(self):
        # torch.nn.init.xavier_uniform_(self.motion_fc1.weight)
        #已弃用
        def xavier_init(m):
            for name,W in m.named_parameters():
                if 'bias' in name:
                    torch.nn.init.constant_(W, 0)
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(W)
        def uniform_init(m):
            for name,W in m.named_parameters():
                if 'bias' in name:
                    torch.nn.init.constant_(W, 0)
                if 'weight' in name:
                    torch.nn.init.uniform_(W)

        def zero_init(m):
            for name,W in m.named_parameters():
                torch.nn.init.constant_(W, 0)

        uniform_init(self.motion_fc2)
        # xavier_init(self.motion_fc2)
        zero_init(self.motion_fc1)
        xavier_init(self.rot_mlp)



    def __init__(self, sh_degree : int,order =16 , rgbfunction="rgbv1"):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._motion = torch.empty(0)
        # self._motion_fourier = torch.empty(0)
        # self.timescale = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._omega = torch.empty(0) #旋转的系数
        # self._color_change = torch.empty(0) #颜色变化的系数

        # self.rgbdecoder = getcolormodel(rgbfuntion)
    
        self.setup_functions()
        self.delta_t = None
        self.omegamask = None 
        self.maskforems = None 
        self.distancetocamera = None
        # self.trbfslinit = None 
        self.ts = None 
        # self.trbfoutput = None 
        self.preprocesspoints = False 
        self.addsphpointsscale = 0.8

        
        self.maxz, self.minz =  0.0 , 0.0 
        self.maxy, self.miny =  0.0 , 0.0 
        self.maxx, self.minx =  0.0 , 0.0  
        self.computedtrbfscale = None
        self.computedopacity = None
        self.raystart = 0.7
        self.duration = 0.0

        # self._ddim = None
        # self._k_neighbors = None
        # self.order = order
        # self.color_order = 0
        self.D=32
        self.H=128
        self.time_emb ,time_out_dim = get_embedder(16,1)
        self.xyz_emb,xyz_out_dim = get_embedder(20,3,include_input=False)
        # self.rot_emb,rot_out_dim = get_embedder(3,4)

        # self.motion_fc1 = nn.Sequential(nn.Linear(self.D+time_out_dim, xyz_out_dim),nn.Tanh()) #encoder
        # self.motion_fc2 = nn.Sequential(nn.Linear(xyz_out_dim, self.H),nn.ReLU(),nn.Linear(self.H, 3)) #decoder
        
        self.motion_fc1 = nn.Sequential(nn.Linear(self.D+time_out_dim, self.H),nn.ReLU(),nn.Linear(self.H,xyz_out_dim)) #encoder
        self.motion_fc2 = nn.Sequential(nn.Linear(xyz_out_dim, self.H),nn.ReLU(),nn.Linear(self.H, self.H),nn.ReLU(),nn.Linear(self.H, 3)) #decoder
        # self.motion_mlp = nn.Sequential(nn.Linear(self.D + time_out_dim+xyz_out_dim,self.H),nn.ReLU(),nn.Linear(self.H, self.H),nn.ReLU(),nn.Linear(self.H, 3))
        self.rot_mlp = nn.Sequential(nn.Linear(self.D+time_out_dim,self.H),nn.ReLU(),nn.Linear(self.H, 4))

        self.bounds = None
        
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    @property
    def get_rotation(self):
        #获取新的旋转
        return self.rotation_activation(self._rotation)
    


    @property
    def get_xyz(self):
        return self._xyz
    @property
    def get_trbfcenter(self):
        return self._trbf_center
    @property
    def get_trbfscale(self):
        return self._trbf_scale
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float,args):
        '''要保证所有的参数都是叶子结点。如果有不是叶子节点的,那么在反传的时候就会报错。nn.parameter就能保证是个叶子节点'''
        if self.preprocesspoints == 3:
            pcd = interpolate_point(pcd, 4) 
        
        elif self.preprocesspoints == 4:
            pcd = interpolate_point(pcd, 2) 
        
        elif self.preprocesspoints == 5:
            pcd = interpolate_point(pcd, 6) 

        elif self.preprocesspoints == 6:
            pcd = interpolate_point(pcd, 8) 
        
        elif self.preprocesspoints == 7:
            pcd = interpolate_point(pcd, 16) 
        
        elif self.preprocesspoints == 8:
            pcd = interpolate_pointv3(pcd, 4) 

        elif self.preprocesspoints == 14:
            pcd = interpolate_partuse(pcd, 2) 
        
        elif self.preprocesspoints == 15:
            pcd = interpolate_partuse(pcd, 4) 

        elif self.preprocesspoints == 16:
            pcd = interpolate_partuse(pcd, 8) 
        
        elif self.preprocesspoints == 17:
            pcd = interpolate_partuse(pcd, 16) 
        else:
            pass 
        self.spatial_lr_scale = spatial_lr_scale
        #random initialiaze
        if args.random_init:
            num_pts = pcd.points.shape[0]
            fused_point_cloud = torch.tensor(np.random.random((num_pts,3))).float().cuda()
            fused_color = torch.tensor(np.random.random((num_pts,3))/255.0).float().cuda()
        else:
            fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        # print(fused_point_cloud)
        # fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        # times = torch.tensor(np.asarray(pcd.times)).float().cuda()


        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        scales = torch.clamp(scales, -10, 1.0)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))


        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)) #[n,1,3]
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)) #[n,15,3]

        N, _ = fused_color.shape

        
        

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))

        omega = torch.zeros((fused_point_cloud.shape[0], self.D), device="cuda") #rotation change encoding
        self._omega = nn.Parameter(omega.requires_grad_(True))
        
        # color_change = torch.zeros((fused_point_cloud.shape[0], self.color_order*3*3), device="cuda") #3个sh，16个多项式basis系数，16个fourierbasis系数
        # self._color_change = nn.Parameter(color_change.requires_grad_(True))

        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        motion = torch.rand((fused_point_cloud.shape[0], self.D), device="cuda")# xyz change encoding,[0,1]uniform distribution
        # motion_fourier = torch.zeros((fused_point_cloud.shape[0], 3*self.order*2), device="cuda") #傅里叶系数
        self._motion = nn.Parameter(motion.requires_grad_(True))
        # self._motion_fourier = nn.Parameter(motion_fourier.requires_grad_(True))
        # self.params_init()
        # self.xavier_init(self.motion_fc1) fc1不需要初始化，需要它初始产生的值较小
        self.xavier_init(self.motion_fc2)
        self.xavier_init(self.rot_mlp)

        self.motion_fc1.to('cuda')
        self.motion_fc2.to('cuda')
        self.rot_mlp.to('cuda')
        # self.motion_mlp.to('cuda')

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # print(torch.ones((self.get_xyz.shape[0], 1), device="cuda").shape)
        # timescale = torch.cat((torch.ones((self.get_xyz.shape[0], 1), device="cuda",requires_grad=True),torch.zeros((self.get_xyz.shape[0], 1), device="cuda",requires_grad=True)),dim=1)
        # self.timescale = nn.Parameter(timescale.requires_grad_(True))
        # print(self.timescale.is_leaf)

        # self._trbf_center = nn.Parameter(times.contiguous().requires_grad_(True))
        # self._trbf_scale = nn.Parameter(torch.ones((self.get_xyz.shape[0], 1), device="cuda").requires_grad_(True)) 

        # ## store gradients


        # if self.trbfslinit is not None:
        #     nn.init.constant_(self._trbf_scale, self.trbfslinit) # too large ? 设置初始值
        # else:
        #     nn.init.constant_(self._trbf_scale, 0) # too large ?

        # nn.init.constant_(self._omega, 0)
        self.mlp_grd = {}


        # print(self._xyz.size())
        self.maxz, self.minz = torch.amax(self._xyz[:,2]), torch.amin(self._xyz[:,2]) 
        self.maxy, self.miny = torch.amax(self._xyz[:,1]), torch.amin(self._xyz[:,1]) 
        self.maxx, self.minx = torch.amax(self._xyz[:,0]), torch.amin(self._xyz[:,0]) 
        self.maxz = min((self.maxz, 200.0)) # some outliers in the n4d datasets.. 
        print(self.bounds,self.maxx, self.maxy, self.maxz, self.minx, self.miny, self.minz)
        # self.bounds = torch.tensor([[self.maxx, self.maxy, self.maxz], [self.minx, self.miny, self.minz]], device="cuda",requires_grad=False)
        
        for name, W in self.motion_fc1.named_parameters():
            self.mlp_grd["motion1"+name] = torch.zeros_like(W, requires_grad=False).cuda()
        for name, W in self.motion_fc2.named_parameters():
            self.mlp_grd["motion2"+name] = torch.zeros_like(W, requires_grad=False).cuda()
        for name, W in self.rot_mlp.named_parameters():
            self.mlp_grd["rot"+name] = torch.zeros_like(W, requires_grad=False).cuda()
        # for name, W in self.motion_mlp.named_parameters():
        #     self.mlp_grd["motion"+name] = torch.zeros_like(W, requires_grad=False).cuda()    

    def cache_gradient(self,stage):
        '''把grad都传给grd'''
        if stage == "dynamatic":#只有dynamtic时，才会有下面这三个的梯度
            self._motion_grd += self._motion.grad.clone()
            # self._motion_fourier_grd += self._motion_fourier.grad.clone()

            for name, W in self.motion_fc1.named_parameters():
                # if 'weight' in name:
                # print(W.grad)
                self.mlp_grd["motion1"+name] = self.mlp_grd["motion1"+name] + W.grad.clone()

        self._xyz_grd += self._xyz.grad.clone()
        self._features_dc_grd += self._features_dc.grad.clone()
        self._scaling_grd += self._scaling.grad.clone()
        # self._rotation_grd += self._rotation.grad.clone()
        self._opacity_grd += self._opacity.grad.clone()
        self._features_rest_grd += self._features_rest.grad.clone()

        self._omega_grd += self._omega.grad.clone()
            
        for name, W in self.rot_mlp.named_parameters():
            self.mlp_grd["rot"+name] = self.mlp_grd["rot"+name] + W.grad.clone()
        # self._trbf_center_grd += self._trbf_center.grad.clone()
        # self._trbf_scale_grd += self._trbf_scale.grad.clone()
        # for name, W in self.motion_mlp.named_parameters():
        #         # if 'weight' in name:
        #     # print(W.grad)
        #     self.mlp_grd["motion"+name] = self.mlp_grd["motion"+name] + W.grad.clone()
        for name, W in self.motion_fc2.named_parameters():
            self.mlp_grd["motion2"+name] = self.mlp_grd["motion2"+name] + W.grad.clone()
        

        # for name, W in self.rgbdecoder.named_parameters():
        #     if 'weight' in name:
        #         self.rgb_grd[name] = self.rgb_grd[name] + W.grad.clone()
    def zero_gradient_cache(self):
        '''把grd都置零'''
        self._xyz_grd = torch.zeros_like(self._xyz, requires_grad=False)
        self._features_dc_grd = torch.zeros_like(self._features_dc, requires_grad=False)
        self._features_rest_grd = torch.zeros_like(self._features_rest, requires_grad=False)

        self._scaling_grd = torch.zeros_like(self._scaling, requires_grad=False)
        self._rotation_grd = torch.zeros_like(self._rotation, requires_grad=False)
        self._opacity_grd = torch.zeros_like(self._opacity, requires_grad=False)
        # self._trbf_center_grd = torch.zeros_like(self._trbf_center, requires_grad=False)
        # self._trbf_scale_grd = torch.zeros_like(self._trbf_scale, requires_grad=False)
        self._motion_grd = torch.zeros_like(self._motion, requires_grad=False)
        # self._motion_fourier_grd = torch.zeros_like(self._motion_fourier, requires_grad=False)

        self._omega_grd = torch.zeros_like(self._omega, requires_grad=False)

        # self.motion_mlp.grd = torch.zeros_like(self.motion_mlp, requires_grad=False)
        # self.rot_mlp.grd = torch.zeros_like(self.rot_mlp, requires_grad=False)

        # self._color_change_grd = torch.zeros_like(self._color_change, requires_grad=False)
        # self.timescale_grd = torch.zeros_like(self.timescale, requires_grad=False)


        for name in self.mlp_grd.keys():
            self.mlp_grd[name].zero_()

    def set_batch_gradient(self, cnt,stage):

        '''把grd通过batch平均后传给grad'''
        ratio = 1/cnt
        self._features_dc.grad = self._features_dc_grd * ratio
        self._features_rest.grad = self._features_rest_grd * ratio
        self._xyz.grad = self._xyz_grd * ratio
        # print(self._xyz.grad)
        self._scaling.grad = self._scaling_grd * ratio
        self._rotation.grad = self._rotation_grd * ratio
        self._opacity.grad = self._opacity_grd * ratio
        # self._trbf_center.grad = self._trbf_center_grd * ratio
        # self._trbf_scale.grad = self._trbf_scale_grd* ratio
        self._motion.grad = self._motion_grd * ratio
        # self._motion_fourier.grad = self._motion_fourier_grd * ratio

        self._omega.grad = self._omega_grd * ratio

        # self.motion_mlp.grad = self.motion_mlp.grd * ratio
        # self.rot_mlp.grad = self.rot_mlp.grd * ratio
        # self._color_change.grad = self._color_change_grd * ratio
        # self.timescale.grad = self.timescale_grd * ratio
        # print("motion_mlp",self.motion_mlp.parameters())

        # for name, W in self.motion_mlp.named_parameters():
        #     W.grad = self.mlp_grd["motion"+name] * ratio
        for name, W in self.motion_fc2.named_parameters():
                # if 'weight' in name:
                    # print(name,W)
            W.grad = self.mlp_grd["motion2"+name] * ratio


        for name, W in self.rot_mlp.named_parameters():
                W.grad = self.mlp_grd["rot"+name] * ratio
        for name, W in self.motion_fc1.named_parameters():

            W.grad = self.mlp_grd["motion1"+name] * ratio

    def training_setup(self, training_args):
        '''设置optimizer'''
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if training_args.use_weight_decay:
            l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._omega], 'lr': training_args.deform_feature_lr,"weight_decay":8e-7,"name": "omega"},
            # {'params': [self._trbf_center], 'lr': training_args.trbfc_lr, "name": "trbf_center"},
            # {'params': [self._trbf_scale], 'lr': training_args.trbfs_lr, "name": "trbf_scale"},
            # {'params': [self._motion], 'lr':  training_args.position_lr_init * self.spatial_lr_scale * 0.5 * training_args.movelr , "name": "motion"},
            # {'params': [self._motion_fourier], 'lr':  training_args.position_lr_init * self.spatial_lr_scale * 0.5 * training_args.movelr , "name": "motion_fourier"}
            {'params': [self._motion], 'lr':   training_args.deform_feature_lr , "weight_decay":8e-7,"name": "motion"},
            # {'params': list(self.motion_mlp.parameters()), 'lr':   training_args.mlp_lr , "weight_decay":8e-7,"name": "motion_mlp"},
            {'params': list(self.rot_mlp.parameters()), 'lr':   training_args.mlp_lr_init , "weight_decay":8e-7,"name": "rot_mlp"},
            {'params': list(self.motion_fc1.parameters()), 'lr':   training_args.mlp_lr_init , "weight_decay":8e-7,"name": "motion_mlp1"},
            {'params': list(self.motion_fc2.parameters()), 'lr':   training_args.mlp_lr , "weight_decay":8e-7,"name": "motion_mlp2"},

            # {'params': [self._motion_fourier], 'lr': training_args.dddm_lr ,"weight_decay":8e-7,"name": "motion_fourier"},
            # {'params': [self._color_change], 'lr': training_args.dddm_lr ,"weight_decay":8e-7,"name": "color_change"},
            # {'params': [self.timescale], 'lr': training_args.dddm_lr ,"weight_decay":8e-7,"name": "timescale"}
                
        ]
        else:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                {'params': [self._omega], 'lr': training_args.deform_feature_lr,"name": "omega"},
                # {'params': [self._trbf_center], 'lr': training_args.trbfc_lr, "name": "trbf_center"},
                # {'params': [self._trbf_scale], 'lr': training_args.trbfs_lr, "name": "trbf_scale"},
                # {'params': [self._motion], 'lr':  training_args.position_lr_init * self.spatial_lr_scale * 0.5 * training_args.movelr , "name": "motion"},
                # {'params': [self._motion_fourier], 'lr':  training_args.position_lr_init * self.spatial_lr_scale * 0.5 * training_args.movelr , "name": "motion_fourier"}
                {'params': [self._motion], 'lr':   training_args.deform_feature_lr , "name": "motion"},
                # {'params': list(self.motion_mlp.parameters()), 'lr':   training_args.mlp_lr , "name": "motion_mlp"},
                {'params': list(self.motion_fc1.parameters()), 'lr':   training_args.mlp_lr , "name": "motion_mlp1"},
                {'params': list(self.motion_fc2.parameters()), 'lr':   training_args.mlp_lr , "name": "motion_mlp2"},

                {'params': list(self.rot_mlp.parameters()), 'lr':   training_args.mlp_lr , "name": "rot_mlp"}
                # {'params': [self._motion_fourier], 'lr': training_args.dddm_lr ,"name": "motion_fourier"},
                # {'params': [self._color_change], 'lr': training_args.dddm_lr ,"name": "color_change"},
                # {'params': [self.timescale], 'lr': training_args.dddm_lr ,"name": "timescale"}
                    
            ]


        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # print(len(self.optimizer.param_groups))
        # print(len(self.optimizer.state))
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.motion_mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_lr_init,
                                                    lr_final=training_args.mlp_lr_final,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        print("move decoder to cuda")
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
    
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                if iteration > 3000:
                    lr = 0
                else:
                    lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # print("xyz",lr)
            if param_group["name"] == "motion_mlp2":
                if iteration > 3000:
                    lr = 0
                elif iteration > 1500:
                    lr = self.motion_mlp_scheduler_args(iteration)
                else:
                    lr = param_group['lr']
                param_group['lr'] = lr
                # print(param_group["name"],lr)
                    # print("motion_mlp",lr)
            if param_group["name"] == "motion":
                param_group['lr'] = 2.5e-2
            if param_group["name"] == "motion_mlp1":
                param_group['lr'] = 2.5e-3
            if param_group["name"] == "f_dc":
                param_group['lr'] = 2.5e-5
            if param_group["name"] == "f_rest":
                param_group['lr'] = 2.5e-6
        # print(self.optimizer.param_groups)

    def construct_list_of_attributes(self):
        # l = ['x', 'y', 'z','trbf_center', 'trbf_scale' ,'nx', 'ny', 'nz'] # 'trbf_center', 'trbf_scale' 
        l = ['x', 'y', 'z' ,'nx', 'ny', 'nz']
        # All channels except the 3 DC
        # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        #     l.append('f_dc_{}'.format(i))
        # for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        #     l.append('f_rest_{}'.format(i))
        for i in range(self._motion.shape[1]):
            l.append('motion_{}'.format(i))
        
        # for i in range(self._motion_fourier.shape[1]):
        #     l.append('motion_fourier_{}'.format(i))

        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._omega.shape[1]):
            l.append('omega_{}'.format(i))
        

        # for i in range(self._color_change.shape[1]):
        #     l.append('color_change_{}'.format(i))
        # for i in range(self.timescale.shape[1]):
        #     l.append('timescale_{}'.format(i))

        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()


        # trbf_center= self._trbf_center.detach().cpu().numpy()

        # trbf_scale = self._trbf_scale.detach().cpu().numpy()
        motion = self._motion.detach().cpu().numpy()
        # motion_fourier = self._motion_fourier.detach().cpu().numpy()

        omega = self._omega.detach().cpu().numpy()
        # motion_mlp = [params.detach().cpu().numpy() for params in self.motion_mlp.parameters()]
        # rot_mlp = [params.detach().cpu().numpy() for params in self.rot_mlp.parameters()]


        # timescale = self.timescale.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, trbf_center, trbf_scale, normals, motion, f_dc, opacities, scale, rotation, omega), axis=1)
        attributes = np.concatenate((xyz, normals,  motion,f_dc,f_rest, opacities, scale, rotation, omega), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        model_fname = path.replace(".ply", ".pth")
        print(f'Saving model checkpoint to: {model_fname}')
        # torch.save(self.rgbdecoder.state_dict(), model_fname)
        torch.save({'motion1_state_dict': self.motion_fc1.state_dict(), 'motion2_state_dict': self.motion_fc2.state_dict(),'rot_state_dict': self.rot_mlp.state_dict()}, model_fname)
        # torch.save({'motion_state_dict': self.motion_mlp.state_dict(),'rot_state_dict': self.rot_mlp.state_dict()}, model_fname)



    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    
    def zero_omega(self, threhold=0.15):
        scales = self.get_scaling
        omegamask = torch.sum(torch.abs(self._omega), dim=1) > threhold # default 
        scalemask = torch.max(scales, dim=1).values.unsqueeze(1) > 0.2
        scalemaskb = torch.max(scales, dim=1).values.unsqueeze(1) < 0.6
        pointopacity = self.get_opacity
        opacitymask = pointopacity > 0.7

        mask = torch.logical_and(torch.logical_and(omegamask.unsqueeze(1), scalemask), torch.logical_and(scalemaskb, opacitymask))
        omeganew = mask.float() * self._omega
        optimizable_tensors = self.replace_tensor_to_optimizer(omeganew, "omega")
        self._omega = optimizable_tensors["omega"]
        return mask
    def zero_omegabymotion(self, threhold=0.15):
        scales = self.get_scaling
        omegamask = torch.sum(torch.abs(self._motion[:, 0:3]), dim=1) > 0.3 #  #torch.sum(torch.abs(self._omega), dim=1) > threhold # default 
        scalemask = torch.max(scales, dim=1).values.unsqueeze(1) > 0.2
        scalemaskb = torch.max(scales, dim=1).values.unsqueeze(1) < 0.6
        pointopacity = self.get_opacity
        opacitymask = pointopacity > 0.7

        

        mask = torch.logical_and(torch.logical_and(omegamask.unsqueeze(1), scalemask), torch.logical_and(scalemaskb, opacitymask)) #根据一系列条件得到 omega的mask
        
        #更新omega
        omeganew = mask.float() * self._omega
        optimizable_tensors = self.replace_tensor_to_optimizer(omeganew, "omega")
        self._omega = optimizable_tensors["omega"]
        return mask


    def zero_omegav2(self, threhold=0.15):
        scales = self.get_scaling
        omegamask = torch.sum(torch.abs(self._omega), dim=1) > threhold # default 
        scalemask = torch.max(scales, dim=1).values.unsqueeze(1) > 0.2
        scalemaskb = torch.max(scales, dim=1).values.unsqueeze(1) < 0.6
        pointopacity = self.get_opacity
        opacitymask = pointopacity > 0.7

        mask = torch.logical_and(torch.logical_and(omegamask.unsqueeze(1), scalemask), torch.logical_and(scalemaskb, opacitymask))
        omeganew = mask.float() * self._omega
        rotationew = self.get_rotation(self.delta_t)


        optimizable_tensors = self.replace_tensor_to_optimizer(omeganew, "omega")
        self._omega = optimizable_tensors["omega"]


        optimizable_tensors = self.replace_tensor_to_optimizer(rotationew, "rotation")
        self._rotation = optimizable_tensors["rotation"]
        return mask

    def load_plyandminmax(self, path,  maxx, maxy, maxz,  minx, miny, minz):
        def logicalorlist(listoftensor):
            mask = None 
            for idx, ele in enumerate(listoftensor):
                if idx == 0 :
                    mask = ele 
                else:
                    mask = np.logical_or(mask, ele)
            return mask 

        plydata = PlyData.read(path)
        #ckpt = torch.load(path.replace(".ply", ".pt"))
        #self.rgbdecoder.load_state_dict(ckpt)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
            #         {'params': [self._trbf_center], 'lr': training_args.trbfc_lr, "name": "trbf_center"},
            # {'params': [self._trbf_scale], 'lr': training_args.trbfs_lr, "name": "trbf_scale"},
        # trbf_center= np.asarray(plydata.elements[0]["trbf_center"])[..., np.newaxis]
        # trbf_scale = np.asarray(plydata.elements[0]["trbf_scale"])[..., np.newaxis]

        #motion = np.asarray(plydata.elements[0]["motion"])
        motion_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("motion")]
        nummotion = 9
        motion = np.zeros((xyz.shape[0], nummotion))
        for i in range(nummotion):
            motion[:, i] = np.asarray(plydata.elements[0]["motion_"+str(i)])


        dc_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_dc")]
        num_dc_features = len(dc_f_names)

        features_dc = np.zeros((xyz.shape[0], num_dc_features))
        for i in range(num_dc_features):
            features_dc[:, i] = np.asarray(plydata.elements[0]["f_dc_"+str(i)])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        #assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], -1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])


        omega_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("omega")]
        omegas = np.zeros((xyz.shape[0], len(omega_names)))
        for idx, attr_name in enumerate(omega_names):
            omegas[:, idx] = np.asarray(plydata.elements[0][attr_name])


        # ft_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_t")]
        # ftomegas = np.zeros((xyz.shape[0], len(ft_names)))
        # for idx, attr_name in enumerate(ft_names):
        #     ftomegas[:, idx] = np.asarray(plydata.elements[0][attr_name])
      

        mask0 = xyz[:,0] > maxx.item()
        mask1 = xyz[:,1] > maxy.item()
        mask2 = xyz[:,2] > maxz.item()

        mask3 = xyz[:,0] < minx.item()
        mask4 = xyz[:,1] < miny.item()
        mask5 = xyz[:,2] < minz.item()
        mask =  logicalorlist([mask0, mask1, mask2, mask3, mask4, mask5])
        mask = np.logical_not(mask)

        
        
        self._xyz = nn.Parameter(torch.tensor(xyz[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        # self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        # self._trbf_center = nn.Parameter(torch.tensor(trbf_center[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        # self._trbf_scale = nn.Parameter(torch.tensor(trbf_scale[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._motion = nn.Parameter(torch.tensor(motion[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._omega = nn.Parameter(torch.tensor(omegas[mask], dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def load_plyandminmaxY(self, path,  maxx, maxy, maxz,  minx, miny, minz):
        def logicalorlist(listoftensor):
            mask = None 
            for idx, ele in enumerate(listoftensor):
                if idx == 0 :
                    mask = ele 
                else:
                    mask = np.logical_or(mask, ele)
            return mask 

        plydata = PlyData.read(path)
        #ckpt = torch.load(path.replace(".ply", ".pt"))
        #self.rgbdecoder.load_state_dict(ckpt)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
            #         {'params': [self._trbf_center], 'lr': training_args.trbfc_lr, "name": "trbf_center"},
            # {'params': [self._trbf_scale], 'lr': training_args.trbfs_lr, "name": "trbf_scale"},
        # trbf_center= np.asarray(plydata.elements[0]["trbf_center"])[..., np.newaxis]
        # trbf_scale = np.asarray(plydata.elements[0]["trbf_scale"])[..., np.newaxis]

        #motion = np.asarray(plydata.elements[0]["motion"])
        motion_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("motion")]
        nummotion = 9
        motion = np.zeros((xyz.shape[0], nummotion))
        for i in range(nummotion):
            motion[:, i] = np.asarray(plydata.elements[0]["motion_"+str(i)])


        dc_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_dc")]
        num_dc_features = len(dc_f_names)

        features_dc = np.zeros((xyz.shape[0], num_dc_features))
        for i in range(num_dc_features):
            features_dc[:, i] = np.asarray(plydata.elements[0]["f_dc_"+str(i)])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        #assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], -1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])


        omega_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("omega")]
        omegas = np.zeros((xyz.shape[0], len(omega_names)))
        for idx, attr_name in enumerate(omega_names):
            omegas[:, idx] = np.asarray(plydata.elements[0][attr_name])


        # ft_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_t")]
        # ftomegas = np.zeros((xyz.shape[0], len(ft_names)))
        # for idx, attr_name in enumerate(ft_names):
        #     ftomegas[:, idx] = np.asarray(plydata.elements[0][attr_name])
      

        #mask0 = xyz[:,0] > maxx.item()
        mask1 = xyz[:,1] > maxy.item()
        #mask2 = xyz[:,2] > maxz.item()

        #mask3 = xyz[:,0] < minx.item()
        mask4 = xyz[:,1] < miny.item()
        #mask5 = xyz[:,2] < minz.item()
        mask =  logicalorlist([mask1 , mask4])
        mask = np.logical_not(mask)

        
        
        self._xyz = nn.Parameter(torch.tensor(xyz[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        # self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        # self._trbf_center = nn.Parameter(torch.tensor(trbf_center[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        # self._trbf_scale = nn.Parameter(torch.tensor(trbf_scale[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._motion = nn.Parameter(torch.tensor(motion[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._omega = nn.Parameter(torch.tensor(omegas[mask], dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree


    def load_plyandminmaxall(self, path,  maxx, maxy, maxz,  minx, miny, minz):
        def logicalorlist(listoftensor):
            mask = None 
            for idx, ele in enumerate(listoftensor):
                if idx == 0 :
                    mask = ele 
                else:
                    mask = np.logical_or(mask, ele)
            return mask 

        plydata = PlyData.read(path)
        #ckpt = torch.load(path.replace(".ply", ".pt"))
        #self.rgbdecoder.load_state_dict(ckpt)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
            #         {'params': [self._trbf_center], 'lr': training_args.trbfc_lr, "name": "trbf_center"},
            # {'params': [self._trbf_scale], 'lr': training_args.trbfs_lr, "name": "trbf_scale"},
        # trbf_center= np.asarray(plydata.elements[0]["trbf_center"])[..., np.newaxis]
        # trbf_scale = np.asarray(plydata.elements[0]["trbf_scale"])[..., np.newaxis]

        #motion = np.asarray(plydata.elements[0]["motion"])
        motion_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("motion")]
        nummotion = 9
        motion = np.zeros((xyz.shape[0], nummotion))
        for i in range(nummotion):
            motion[:, i] = np.asarray(plydata.elements[0]["motion_"+str(i)])


        dc_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_dc")]
        num_dc_features = len(dc_f_names)

        features_dc = np.zeros((xyz.shape[0], num_dc_features))
        for i in range(num_dc_features):
            features_dc[:, i] = np.asarray(plydata.elements[0]["f_dc_"+str(i)])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        #assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], -1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])


        omega_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("omega")]
        omegas = np.zeros((xyz.shape[0], len(omega_names)))
        for idx, attr_name in enumerate(omega_names):
            omegas[:, idx] = np.asarray(plydata.elements[0][attr_name])


        # ft_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_t")]
        # ftomegas = np.zeros((xyz.shape[0], len(ft_names)))
        # for idx, attr_name in enumerate(ft_names):
        #     ftomegas[:, idx] = np.asarray(plydata.elements[0][attr_name])
      

        mask0 = xyz[:,0] > maxx.item()
        mask1 = xyz[:,1] > maxy.item()
        mask2 = xyz[:,2] > maxz.item()

        mask3 = xyz[:,0] < minx.item()
        mask4 = xyz[:,1] < miny.item()
        mask5 = xyz[:,2] < minz.item()
        mask =  logicalorlist([mask0, mask1, mask2, mask3, mask4, mask5])
        #mask = np.logical_not(mask)# now the reset point is within the boundray

        unstablepoints = np.sum(np.abs(motion[:, 0:3]),axis=1) 
        movingpoints = unstablepoints > 0.03
        # trbfmask = trbf_scale < 3 # temporal unstable points

        # maskst = np.logical_or(trbfmask.squeeze(1), movingpoints)

        # mask = np.logical_or(mask, maskst) # only use large tscale points.
        # replace points with input ?

        mask  = np.logical_not(mask)# remaining good points. todo remove good mask's NN 

        xyz = torch.cat((self._xyz, torch.tensor(xyz[mask], dtype=torch.float, device="cuda")))
        
        self._xyz = nn.Parameter(xyz.requires_grad_(True))

        features_dc= torch.cat((self._features_dc, torch.tensor(features_dc[mask], dtype=torch.float, device="cuda")))
        self._features_dc = nn.Parameter(features_dc.requires_grad_(True))

        opacities = torch.cat((self._opacity, torch.tensor(opacities[mask], dtype=torch.float, device="cuda")))
        self._opacity = nn.Parameter(opacities).requires_grad_(True)

        scales = torch.cat((self._scaling, torch.tensor(scales[mask], dtype=torch.float, device="cuda")))

        self._scaling = nn.Parameter(scales).requires_grad_(True)
        rots = torch.cat((self._rotation, torch.tensor(rots[mask], dtype=torch.float, device="cuda")))

        self._rotation = nn.Parameter(rots).requires_grad_(True)
        # trbf_center =  torch.cat((self._trbf_center, torch.tensor(trbf_center[mask], dtype=torch.float, device="cuda")))
        # self._trbf_center = nn.Parameter(trbf_center).requires_grad_(True)
        # trbf_scale =  torch.cat((self._trbf_scale, torch.tensor(trbf_scale[mask], dtype=torch.float, device="cuda")))


        # self._trbf_scale = nn.Parameter(trbf_scale.requires_grad_(True))

        motion =  torch.cat((self._motion, torch.tensor(motion[mask], dtype=torch.float, device="cuda")))

        self._motion = nn.Parameter(motion.requires_grad_(True))
        omegas = torch.cat((self._omega, torch.tensor(omegas[mask], dtype=torch.float, device="cuda")))
        self._omega = nn.Parameter(omegas.requires_grad_(True))


        self.active_sh_degree = self.max_sh_degree
    def load_ply(self, path):
        print(path)
        plydata = PlyData.read(path)
        #ckpt = torch.load(path.replace(".ply", ".pt"))
        #self.rgbdecoder.load_state_dict(ckpt)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
            #         {'params': [self._trbf_center], 'lr': training_args.trbfc_lr, "name": "trbf_center"},
            # {'params': [self._trbf_scale], 'lr': training_args.trbfs_lr, "name": "trbf_scale"},
        # trbf_center= np.asarray(plydata.elements[0]["trbf_center"])[..., np.newaxis]
        # trbf_scale = np.asarray(plydata.elements[0]["trbf_scale"])[..., np.newaxis]
        

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        #motion = np.asarray(plydata.elements[0]["motion"])
        motion_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("motion")]
        
        nummotion = self.D
        motion = np.zeros((xyz.shape[0], nummotion))
        # motion_fourier = np.zeros((xyz.shape[0], nummotion_fourier))
        for i in range(nummotion):
            motion[:, i] = np.asarray(plydata.elements[0]["motion_"+str(i)])
        # motion = np.random.random((xyz.shape[0], nummotion)) #only for test
        # for i in range(nummotion_fourier):
        #     motion_fourier[:, i] = np.asarray(plydata.elements[0]["motion_fourier_"+str(i)])

        # dc_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_dc")]
        # num_dc_features = len(dc_f_names)

        # features_dc = np.zeros((xyz.shape[0], num_dc_features))
        # for i in range(num_dc_features):
        #     features_dc[:, i] = np.asarray(plydata.elements[0]["f_dc_"+str(i)])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

        #assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        # features_extra = features_extra.reshape((features_extra.shape[0], -1))
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])


        omega_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("omega")]
        omegas = np.zeros((xyz.shape[0], len(omega_names)))
        for idx, attr_name in enumerate(omega_names):
            omegas[:, idx] = np.asarray(plydata.elements[0][attr_name])

        mlp_state_dict = torch.load(path.replace(".ply", ".pth"))
        # color_change_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("color_change")]
        # color_change = np.zeros((xyz.shape[0], len(color_change_names)))
        # for idx, attr_name in enumerate(color_change_names):
        #     color_change[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # time_scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("timescale")]
        # timescale = np.zeros((xyz.shape[0], len(time_scale_names)))
        # for idx, attr_name in enumerate(time_scale_names):
        #     timescale[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # ft_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_t")]
        # ftomegas = np.zeros((xyz.shape[0], len(ft_names)))
        # for idx, attr_name in enumerate(ft_names):
        #     ftomegas[:, idx] = np.asarray(plydata.elements[0][attr_name])



        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._trbf_center = nn.Parameter(torch.tensor(trbf_center, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._trbf_scale = nn.Parameter(torch.tensor(trbf_scale, dtype=torch.float, device="cuda").requires_grad_(True))
        self._motion = nn.Parameter(torch.tensor(motion, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._motion_fourier = nn.Parameter(torch.tensor(motion_fourier, dtype=torch.float, device="cuda").requires_grad_(True))

        self._omega = nn.Parameter(torch.tensor(omegas, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._color_change = nn.Parameter(torch.tensor(color_change, dtype=torch.float, device="cuda").requires_grad_(True))
        # print(self._color_change.shape)
        # self.timescale = nn.Parameter(torch.tensor(timescale, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree
        # self.computedtrbfscale = torch.exp(self._trbf_scale) # precomputed
        self.computedopacity =self.opacity_activation(self._opacity)
        self.computedscales = torch.exp(self._scaling) # change not very large

        # self.motion_mlp.load_state_dict({k.replace('motion_', ''): v for k, v in mlp_state_dict['motion_state_dict'].items()})
        self.motion_fc1.load_state_dict({k.replace('motion1_', ''): v for k, v in mlp_state_dict['motion1_state_dict'].items()})
        self.motion_fc2.load_state_dict({k.replace('motion2_', ''): v for k, v in mlp_state_dict['motion2_state_dict'].items()})
        self.rot_mlp.load_state_dict({k.replace('rot_', ''): v for k, v in mlp_state_dict['rot_state_dict'].items()})
        
        # self.constant_init(self.motion_fc1,0)#only for test
        
        self.motion_fc1.to("cuda")
        self.motion_fc2.to("cuda")
        self.rot_mlp.to("cuda")
        # self.motion_mlp.to("cuda")

        self.mlp_grd = {}
        for name, W in self.motion_fc1.named_parameters():
            self.mlp_grd["motion1"+name] = torch.zeros_like(W, requires_grad=False).cuda()
        for name, W in self.motion_fc2.named_parameters():
            self.mlp_grd["motion2"+name] = torch.zeros_like(W, requires_grad=False).cuda()
        for name, W in self.rot_mlp.named_parameters():
            self.mlp_grd["rot"+name] = torch.zeros_like(W, requires_grad=False).cuda()
        # for name, W in self.motion_mlp.named_parameters():
        #     self.mlp_grd["motion"+name] = torch.zeros_like(W, requires_grad=False).cuda()

    def replace_tensor_to_optimizer(self, tensor, name):
        '''将optim中对应name的值给换成tensor，并且adam中原本保存的状态清0'''
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # print(group,len(group["params"]))
            if len(group["params"]) == 1 and group["name"] != 'decoder' :
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True))) #生成的是leaf节点
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        # self._trbf_center = optimizable_tensors["trbf_center"]
        # self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._motion = optimizable_tensors["motion"]
        self._omega = optimizable_tensors["omega"]
        # self._color_change = optimizable_tensors["color_change"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        # print(self.max_radii2D.device)
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        # self._motion_fourier = optimizable_tensors["motion_fourier"]
        # self.timescale = optimizable_tensors["timescale"]
        if self.omegamask is not None :
            self.omegamask = self.omegamask[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        '''将tensors cat 到optimizer中，即加入进去'''
        optimizable_tensors = {}
        # print(len(self.optimizer.param_groups))
        # print(len(self.optimizer.state))
        for group in self.optimizer.param_groups:
            if len(group["params"]) == 1 and group["name"] in tensors_dict:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_feature_rest ,new_opacities, new_scaling, new_rotation, new_motion, new_omega, dummy=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_feature_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        # "trbf_center" : new_trbf_center,
        # "trbf_scale" : new_trbfscale,
        "motion": new_motion,
        "omega": new_omega}

        optimizable_tensors = self.cat_tensors_to_optimizer(d) #将这些点加入进去
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        # self._trbf_center = optimizable_tensors["trbf_center"]
        # self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._motion = optimizable_tensors["motion"]
        self._omega = optimizable_tensors["omega"]
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    

    def densify_and_splitv2(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        #trbfmask =  self.trbfoutput > 0.8
        #selected_pts_mask = torch.logical_and(selected_pts_mask, trbfmask.squeeze(1))
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1) # n,1,1 to n1
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        # new_trbf_center = self._trbf_center[selected_pts_mask].repeat(N,1)
        # new_trbf_center = torch.rand_like(new_trbf_center) #* 0.5
        # new_trbf_scale = self._trbf_scale[selected_pts_mask].repeat(N,1)
        new_motion = self._motion[selected_pts_mask].repeat(N,1)
        new_omega = self._omega[selected_pts_mask].repeat(N,1)

        # self.densification_postfix(new_xyz, new_features_dc, new_opacity, new_scaling, new_rotation, new_trbf_center, new_trbf_scale, new_motion, new_omega)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,new_opacity, new_scaling, new_rotation,  new_motion, new_omega)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
    



    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        #trbfmask =  self.trbfoutput > 0.8
        #selected_pts_mask = torch.logical_and(selected_pts_mask, trbfmask.squeeze(1))
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        #new_trbf_center =  torch.rand((self._trbf_center[selected_pts_mask].shape[0], 1), device="cuda")  #self._trbf_center[selected_pts_mask]
        #new_trbf_center =  self._trbf_center[selected_pts_mask] # 
        # new_trbf_center =  torch.rand((self._trbf_center[selected_pts_mask].shape[0], 1), device="cuda")  #self._trbf_center[selected_pts_mask]
        # new_trbfscale = self._trbf_scale[selected_pts_mask]
        new_motion = self._motion[selected_pts_mask]

        new_omega = self._omega[selected_pts_mask]

        #self.trbfoutput = torch.cat((self.trbfoutput, torch.zeros(N , 1).to(self.trbfoutput)))
        # self.densification_postfix(new_xyz, new_features_dc, new_opacities, new_scaling, new_rotation, new_trbf_center, new_trbfscale, new_motion, new_omega)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,new_opacities, new_scaling, new_rotation,  new_motion, new_omega)



    def densify_pruneclone(self, max_grad, min_opacity, extent, max_screen_size, splitN=1):
        #print("before", torch.amax(self.get_scaling))
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        with torch.no_grad():
            print(torch.mean(grads), torch.amax(grads), torch.amin(grads))
            print(torch.mean(self.get_opacity), torch.amax(self.get_opacity), torch.amin(self.get_opacity))

        print("befre clone", self._xyz.shape[0])
        self.densify_and_clone(grads, max_grad, extent)
        print("after clone", self._xyz.shape[0])

        self.densify_and_splitv2(grads, max_grad, extent, 2)
        print("after split", self._xyz.shape[0])

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size  
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def addgaussians(self, baduvidx, viewpoint_cam, depthmap, gt_image, numperay=3, ratioend=2, trbfcenter=0.5,depthmax=None,shuffle=False):
        def pix2ndc(v, S):
            return (v * 2.0 + 1.0) / S - 1.0
        ratiaolist = torch.linspace(self.raystart, ratioend, numperay) # 0.7 to ratiostart
        rgbs = gt_image[:, baduvidx[:,0], baduvidx[:,1]]
        rgbs = rgbs.permute(1,0)
        featuredc = rgbs #torch.cat((rgbs, torch.zeros_like(rgbs)), dim=1)# should we add the feature dc with non zero values?

        depths = depthmap[:, baduvidx[:,0], baduvidx[:,1]]
        depths = depths.permute(1,0) # only use depth map > 15 .

        #maxdepth = torch.amax(depths) # not use max depth, use the top 5% depths? avoid to much growng
        depths = torch.ones_like(depths) * depthmax # use the max local depth for the scene ?

        
        u = baduvidx[:,0] # hight y
        v = baduvidx[:,1] # weidth  x 
        #v =  viewpoint_cam.image_height - v 
        Npoints = u.shape[0]
          
        new_xyz = []
        new_scaling = []
        new_rotation = []
        new_features_dc = []
        new_opacity = []
        new_trbf_center = []
        new_trbf_scale = []

        new_motion = []
        new_motion_fourier = []

        new_omega = []
        new_featuret = [ ]

        camera2wold = viewpoint_cam.world_view_transform.T.inverse()
        projectinverse = viewpoint_cam.projection_matrix.T.inverse()
        maxz, minz = self.maxz, self.minz 
        maxy, miny = self.maxy, self.miny 
        maxx, minx = self.maxx, self.minx  
        

        for zscale in ratiaolist :
            ndcu, ndcv = pix2ndc(u, viewpoint_cam.image_height), pix2ndc(v, viewpoint_cam.image_width)
            # targetPz = depths*zscale # depth in local cameras..
            if shuffle == True:
                randomdepth = torch.rand_like(depths) - 0.5 # -0.5 to 0.5
                targetPz = (depths + depths/10*(randomdepth)) *zscale 
            else:
                targetPz = depths*zscale # depth in local cameras..
            
            ndcu = ndcu.unsqueeze(1)
            ndcv = ndcv.unsqueeze(1)


            ndccamera = torch.cat((ndcv, ndcu,   torch.ones_like(ndcu) * (1.0) , torch.ones_like(ndcu)), 1) # N,4 ...
            
            localpointuv = ndccamera @ projectinverse.T 

            diretioninlocal = localpointuv / localpointuv[:,3:] # ray direction in camera space 


            rate = targetPz / diretioninlocal[:, 2:3] #  
            
            localpoint = diretioninlocal * rate

            localpoint[:, -1] = 1
            
            
            worldpointH = localpoint @ camera2wold.T  #myproduct4x4batch(localpoint, camera2wold) # 
            worldpoint = worldpointH / worldpointH[:, 3:] #  

            xyz = worldpoint[:, :3] 
            distancetocameracenter = viewpoint_cam.camera_center - xyz
            distancetocameracenter = torch.norm(distancetocameracenter, dim=1)

            xmask = torch.logical_and(xyz[:, 0] > minx, xyz[:, 0] < maxx )
            #ymask = torch.logical_and(xyz[:, 1] > miny, xyz[:, 1] < maxy )
            #zmask = torch.logical_and(xyz[:, 2] > minz, xyz[:, 2] < maxz )
            #selectedmask = torch.logical_and(xmask,torch.logical_and(ymask,zmask))
            selectedmask = torch.logical_or(xmask, torch.logical_not(xmask))  #torch.logical_and(xmask, ymask)
            new_xyz.append(xyz[selectedmask]) 
            #new_xyz.append(xyz) 

            #new_scaling.append(newscalingmean.repeat(Npoints,1))
            #new_rotation.append(newrotationmean.repeat(Npoints,1))
            new_features_dc.append(featuredc.cuda(0)[selectedmask])
            #new_opacity.append(newopacitymean.repeat(Npoints,1))
            
#            new_trbf_center.append(torch.rand(1).cuda() * torch.ones((Npoints, 1), device="cuda")) 
            selectnumpoints = torch.sum(selectedmask).item()
            # new_trbf_center.append(torch.rand((selectnumpoints, 1)).cuda())

            assert self.trbfslinit < 1 
            # new_trbf_scale.append(self.trbfslinit * torch.ones((selectnumpoints, 1), device="cuda"))
            new_motion.append(torch.zeros((selectnumpoints, 3*16), device="cuda")) 
            new_motion_fourier.append(torch.zeros((selectnumpoints, 3*16*2), device="cuda"))

            new_omega.append(torch.zeros((selectnumpoints, 4), device="cuda"))
            new_featuret.append(torch.zeros((selectnumpoints, 3), device="cuda"))

        new_xyz = torch.cat(new_xyz, dim=0)
        #new_scaling = torch.cat(new_scaling, dim=0)
        #new_rotation = torch.cat(new_rotation, dim=0)
        new_rotation = torch.zeros((new_xyz.shape[0],4), device="cuda")
        new_rotation[:, 1]= 0
        
        new_features_dc = torch.cat(new_features_dc, dim=0)
        #new_opacity = torch.cat(new_opacity, dim=0)
        new_opacity = inverse_sigmoid(0.1 *torch.ones_like(new_xyz[:, 0:1]))
        # new_trbf_center = torch.cat(new_trbf_center, dim=0)
        # new_trbf_scale = torch.cat(new_trbf_scale, dim=0)
        new_motion = torch.cat(new_motion, dim=0)
        new_motion_fourier = torch.cat(new_motion_fourier, dim=0)

        new_omega = torch.cat(new_omega, dim=0)
        new_featuret = torch.cat(new_featuret, dim=0)

         

        tmpxyz = torch.cat((new_xyz, self._xyz), dim=0)
        dist2 = torch.clamp_min(distCUDA2(tmpxyz), 0.0000001)
        dist2 = dist2[:new_xyz.shape[0]]
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        scales = torch.clamp(scales, -10, 1.0)
        new_scaling = scales 


        # self.densification_postfix(new_xyz, new_features_dc, new_opacity, new_scaling, new_rotation, new_trbf_center, new_trbf_scale, new_motion, new_omega,new_featuret)
        self.densification_postfix(new_xyz, new_features_dc, new_opacity, new_scaling, new_rotation, new_motion, new_motion_fourier,new_omega,new_featuret)
        return new_xyz.shape[0]


    def prune_pointswithemsmask(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        #self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._trbf_center = optimizable_tensors["trbf_center"]
        self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._motion = optimizable_tensors["motion"]
        self._omega = optimizable_tensors["omega"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self.maskforems = self.maskforems[valid_points_mask] # we only remain valid mask from ems 



    def reducepointsopscalebymask(self, selected_pts_mask):
        # restrict orignal one
        meanvalue = torch.min(self.get_opacity)
        opacities_new = inverse_sigmoid( torch.min( torch.ones_like(self.get_opacity[selected_pts_mask])*meanvalue , torch.ones_like(self.get_opacity[selected_pts_mask])*0.004))
        opacityold = self._opacity.clone()
        opacityold[selected_pts_mask] = opacities_new


        optimizable_tensors = self.replace_tensor_to_optimizer(opacityold, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def get_motion(self, timestamp):
        time_embbed = self.time_emb(torch.tensor([timestamp],dtype=torch.float, device="cuda"))
        # print(time_embbed)
        # print(self._xyz)
        xyz_embbed = self.xyz_emb(self.normalize_xyz)
        time_embbed = time_embbed.repeat(self._motion.shape[0],1)
        motion_input_1 = torch.cat((self._motion,time_embbed),dim=1)
        encoding = self.motion_fc1(motion_input_1)
        motion_input_2 = xyz_embbed + encoding
        motion_out = self.inv_normalize_xyz(torch.sigmoid(self.motion_fc2(motion_input_2)))

        return motion_out

    def get_rot(self, timestamp):
        time_embbed = self.time_emb(torch.tensor([timestamp],dtype=torch.float, device="cuda"))
        time_embbed = time_embbed.repeat(self._motion.shape[0],1)

        rot_input = torch.cat((self._omega,time_embbed),dim=1)
        rot = self.rotation_activation(self.rot_mlp(rot_input))
        return rot
    def get_motion_l1loss(self):
        return torch.mean(torch.norm(self._motion, p=1,dim=1)) + torch.mean(torch.norm(self._motion_fourier, p=1,dim=1))

    # def adptive_timescale(self,t):
    #     '''return: [N,]'''
    #     return self.timescale@torch.tensor([t,1.],dtype=torch.float, device="cuda",requires_grad=False).unsqueeze(1) #N,2 @ 2,
    
    def get_time_smoothloss(self,t):
        # print(self._ddim)
        if self._ddim is None: #表示还没有计算过ddim，即还没有开始动态阶段
            return 0
        current_ddim = self._ddim
        delta_t = t + 0.1/self.duration
        _,_,_,next_ddim = self.get_ddim(delta_t,frame_time=False)
        # next_ddim = torch.cat((next_motion,next_rot,next_color),dim=1) #[N,9]
        smoothloss = torch.norm(current_ddim - next_ddim, p=2,dim=1).mean()
        # print(smoothloss)
        return smoothloss

    def get_rot_l1loss(self):
        return torch.mean(torch.norm(self._omega, p=1,dim=1))
    
    # def get_ddim(self,t,frame_time=True):

        
    #     time_per_gs = self.adptive_timescale(t)
        

    #     time_order_list=[torch.pow(time_per_gs,i+1) for i in range(self.order)]
    #     poly_basis = torch.stack(time_order_list,dim=1).squeeze() #[N,order]
    #     fourier_basis = time_per_gs * torch.tensor([i+1 for i in range(self.order)],dtype=torch.float, device="cuda") #[N,order]
    #     fourier_basis = torch.stack([torch.sin(fourier_basis),torch.cos(fourier_basis)] ,dim=-1) #[N,order,2]

    #     motion_poly_coef = self._motion.reshape(self._xyz.shape[0],3,-1) #[N,3,order]
    #     motion_fourier_coef = self._motion_fourier.reshape(self._xyz.shape[0],3,-1,2) #[N,3,order,2]

    #     omega = self._omega.reshape(self._xyz.shape[0],3,-1) #[N,3,order]
    #     rot_poly_coef = omega[:,0].reshape(self._xyz.shape[0],4,-1) #[N,3,order]
    #     rot_fourier_coef = omega[:,1:].reshape(self._xyz.shape[0],4,-1,2) #[N,3,order,2]

    #     color_change = self._color_change.reshape(self._xyz.shape[0],3,-1)
    #     color_poly_coef = color_change[:,0].reshape(self._xyz.shape[0],3,-1)
    #     color_fourier_coef = color_change[:,1:].reshape(self._xyz.shape[0],3,-1,2)
    #     # motion = self.get_xyz + (motion_poly_coef * poly_basis.unsqueeze(1)).sum(-1) + (motion_fourier_coef * fourier_basis.unsqueeze(1)).sum([-1,-2])

    #     motion = self.get_xyz + (motion_poly_coef * poly_basis.unsqueeze(1)).sum(-1) + (motion_fourier_coef * fourier_basis.unsqueeze(1)).sum([-1,-2])
    #     rot = self._rotation + (rot_poly_coef * poly_basis.unsqueeze(1)).sum(-1) + (rot_fourier_coef * fourier_basis.unsqueeze(1)).sum([-1,-2])
    #     # print(self.color_order)
    #     # print(self._features_dc.shape,color_poly_coef.shape,color_fourier_coef.shape,poly_basis.shape,fourier_basis.shape)
    #     color_dc = self._features_dc.squeeze()+ (color_poly_coef * poly_basis[:,:self.color_order].unsqueeze(1)).sum(-1) + (color_fourier_coef * fourier_basis[:,:self.color_order].unsqueeze(1)).sum([-1,-2])
        
    #     color = torch.cat((color_dc.unsqueeze(1),self._features_rest),dim=1)
    #     ddim=torch.cat((motion,rot,color_dc),dim=1)
    #     if frame_time:
    #         self._ddim =ddim #如果现在是帧内时间,将ddim记录下来，在计算loss时可以用
    #     return motion, rot, color ,ddim
        
    def get_rigid_loss(self):
        if self._ddim is None:
            return 0
        if self._k_neighbors is None:
            kdtree = KDTree(self.get_xyz.detach().cpu().numpy())
            self._k_neighbors = kdtree.query(self.get_xyz.detach().cpu().numpy(), k=10, return_distance=False)
            print("kdtree builded")
        a= self._ddim[self._k_neighbors[:,1:]]
        b = self._ddim[self._k_neighbors[:,0]]
        # print(a.shape,b.shape)
        k_neighbors_ddim = self._ddim[self._k_neighbors[:,1:]] -self._ddim[self._k_neighbors[:,0]].unsqueeze(1) #[N,9,3+3+4]
        rigid_loss = torch.linalg.vector_norm(k_neighbors_ddim, ord=2,dim=2).sum(dim=1).mean()
        return rigid_loss

    @property
    def get_points_num(self):
        return self._xyz.shape[0]

    def get_deformation(self,timestamp,stage):
        '''返回的motion_out为mlp的输出结果'''
        time_embbed = self.time_emb(torch.tensor([timestamp],dtype=torch.float, device="cuda"))
        # print(time_embbed)
        # print(self._xyz)
        xyz_embbed = self.xyz_emb(self.normalize_xyz)
        time_embbed = time_embbed.repeat(self._motion.shape[0],1)
        if stage == "static":
            # zero_feature = torch.zeros((self._xyz.shape[0],(time_embbed.shape[1]+self.D)),device="cuda")
            motion_input = xyz_embbed.detach()
            # print(motion_input.shape)
            motion_out = self.motion_fc2(motion_input)
            # print(motion_out)
            motion_out = torch.sigmoid(motion_out)
            # print(motion_out)
            # print(self.normalize_xyz(),motion_out)
            self.static_xyz = motion_out

            motion_out = self.inv_normalize_xyz(motion_out)

            motion = self._xyz #static阶段仍然用这个作为motion
            # print(motion.shape)
            # rot = self.rotation_activation(self._rotation)
        elif stage == "static_mlp":
            # print(self.motion_mlp.state_dict())
            # exit()
            # zero_feature = torch.zeros((self._xyz.shape[0],(time_embbed.shape[1]+self.D)),device="cuda")
            motion_input = xyz_embbed
            # motion_input = xyz_embbed
            motion_out = self.motion_fc2(motion_input)
            # print(motion_out)
            motion_out = torch.sigmoid(motion_out)
            # print(self.normalize_xyz(),motion_out)
            self.static_xyz = motion_out
            motion_out = self.inv_normalize_xyz(motion_out)
            
            motion = motion_out #static阶段仍然用这个作为motion
            # rot = self.rotation_activation(self._rotation)
            # print(self._xyz,motion)
            # print(self.motion_mlp.state_dict())
            # exit()

        else:
            # motion_feature = torch.cat(self.motion_fc1(torch.cat((self._motion,time_embbed),dim=1)))
            motion_input_1 = torch.cat((self._motion,time_embbed),dim=1)
            encoding = self.motion_fc1(motion_input_1)
            # for name,W in self.motion_fc1.named_parameters():
            #     print(name,W)
            # print(self._motion,time_embbed)
            # print(encoding)
            motion_input_2 = xyz_embbed + encoding
            # print(xyz_embbed,motion_input_2)
            motion_out = self.inv_normalize_xyz(torch.sigmoid(self.motion_fc2(motion_input_2)))
            motion = motion_out
            # print(self._xyz,motion)
            # exit()


        rot_input = torch.cat((self._omega,time_embbed),dim=1)
        rot = self.rotation_activation(self.rot_mlp(rot_input))
        # print(self._xyz,motion)
        # print(self._motion)
        # print(torch.mean(motion,dim=0))
        # print(torch.var(motion,dim=0))

        return motion, rot ,motion_out
    @property
    def normalize_xyz(self):
        min_bounds = self.bounds[1]
        max_bounds = self.bounds[0]
        return (self._xyz - min_bounds) / (max_bounds - min_bounds)

    def inv_normalize_xyz(self,norm_xyz):
        min_bounds = self.bounds[1]
        max_bounds = self.bounds[0]
        return norm_xyz * (max_bounds - min_bounds) + min_bounds

    def set_bounds(self,xyz_max, xyz_min):
        bounds = torch.tensor([xyz_max, xyz_min],dtype=torch.float32,device='cuda')
        self.bounds = nn.Parameter(bounds,requires_grad=False)

def get_embedder(multires,input_dims ,include_input=True,i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2'] #L-1
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

