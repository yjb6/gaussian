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
from scene.triplane import TriPlaneField
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
        #self.featureact = torch.sigmoid

    def params_init(self):
        torch.nn.init.xavier_uniform_(self.motion_fc1.weight)
        def xavier_init(m):
            for name,W in m.named_parameters():
                if 'bias' in name:
                    torch.nn.init.constant_(W, 1)
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(W)
        xavier_init(self.motion_mlp)
        xavier_init(self.rot_mlp)
        # torch.nn.init.xavier_uniform_(self.motion_fc2.weight)



    def __init__(self, args,order =16 , rgbfunction="rgbv1"):
        self.args=args
        # self.args.dynamatic_mlp= False
        self.active_sh_degree = 0
        self.max_sh_degree = args.sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        # self._motion = torch.empty(0)
        # self._motion_fourier = torch.empty(0)
        # self.timescale = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        # self._omega = torch.empty(0) #旋转的系数
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
        self.D=args.deform_feature_dim #16
        self.H=args.deform_hidden_dim #128
        self.time_emb ,out_dim = get_embedder(args.deform_time_encode) #4
        if hasattr(self.args, 'multiscale_time'):
            multiscale_time = self.args.multiscale_time
        else:
            multiscale_time = False
        if not hasattr(self.args,"dsh"):
            self.args.dsh = False
        if not hasattr(self.args,"abtest1"):
            self.args.abtest1 = False
        self.key_frame_nums = self.args.key_frame_nums

        self.deform_field =nn.ModuleList([TriPlaneField(args.bounds, args.kplanes_config, args.multires,multiscale_time=multiscale_time) for i in range(self.key_frame_nums)])
        # self.forward_field = TriPlaneField(args.bounds, args.kplanes_config, args.multires,multiscale_time=multiscale_time)
        # self.backward_field = TriPlaneField(args.bounds, args.kplanes_config, args.multires,multiscale_time=multiscale_time)
        hexplane_outdim = self.deform_field[0].feat_dim

        self.motion_mlp = nn.Sequential(nn.Linear(hexplane_outdim+out_dim,self.H),nn.ReLU(),nn.Linear(self.H,self.H),nn.ReLU(),nn.Linear(self.H, 3))
        # self.motion_mlp = nn.Sequential(nn.Linear(self.D + out_dim,self.H),nn.ReLU(),nn.Linear(self.H, self.H),nn.ReLU(),nn.Linear(self.H, 3))
        self.rot_mlp = nn.Sequential(nn.Linear(hexplane_outdim+out_dim,self.H),nn.ReLU(),nn.Linear(self.H,self.H),nn.ReLU(),nn.Linear(self.H, 7))
        # print(self.H/2)
        self.opacity_mlp = nn.Sequential(nn.Linear(hexplane_outdim+out_dim,self.H),nn.ReLU(),nn.Linear(self.H,int(self.H/2) ),nn.ReLU(),nn.Linear(int(self.H/2), 1))#考虑整合进某个别的mlp中
        # self.args.dsh =False

        if self.args.dsh:
            self.sh_mlp = nn.Sequential(nn.Linear(hexplane_outdim+out_dim,self.H),nn.ReLU(),nn.Linear(self.H,self.H),nn.ReLU(),nn.Linear(self.H, 3))
        if self.args.dynamatic_mlp:
            self.dynamatic_mlp = nn.Sequential(nn.Linear(self.D+out_dim+hexplane_outdim,self.H),nn.ReLU(),nn.Linear(self.H,2),nn.Softmax(dim=1))
            self.dynamatic = torch.empty(0)
        # print(self.forward_field,self.backward_field,hexplane_outdim)

        # self.start_end = torch.empty(0)
        assert self.key_frame_nums > 2 #至少要有三个关键帧
        self.is_dynamatic =False
        self.valid_mask =None
        # self.end = torch.empty(0)

        
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
            pcd = interpolate_point(pcd, 40) #0.025 
        
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
            #  == 17:
            pcd = interpolate_partuse(pcd, self.preprocesspoints)
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        # print(fused_point_cloud)
        # fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        # times = torch.tensor(np.asarray(pcd.times)).float().cuda()


        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
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

        # omega = torch.zeros((fused_point_cloud.shape[0], self.D), device="cuda") #rotation change encoding
        # self._omega = nn.Parameter(omega.requires_grad_(True))
        
        # color_change = torch.zeros((fused_point_cloud.shape[0], self.color_order*3*3), device="cuda") #3个sh，16个多项式basis系数，16个fourierbasis系数
        # self._color_change = nn.Parameter(color_change.requires_grad_(True))

        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        # motion = torch.zeros((fused_point_cloud.shape[0], self.D), device="cuda")# xyz change encoding
        # motion_fourier = torch.zeros((fused_point_cloud.shape[0], 3*self.order*2), device="cuda") #傅里叶系数
        # self._motion = nn.Parameter(motion.requires_grad_(True))
        # self._motion_fourier = nn.Parameter(motion_fourier.requires_grad_(True))
        self.motion_mlp.to('cuda')
        self.rot_mlp.to('cuda')
        self.deform_field.to('cuda')
        # self.forward_field.to('cuda')
        # self.backward_field.to('cuda')
        self.opacity_mlp.to('cuda')
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")




        # if self.trbfslinit is not None:
        #     nn.init.constant_(self._trbf_scale, self.trbfslinit) # too large ? 设置初始值
        # else:
        #     nn.init.constant_(self._trbf_scale, 0) # too large ?

        # nn.init.constant_(self._omega, 0)
        self.mlp_grd = {}
        self.init_mlp_grd()

        # print(self._xyz.size())
        self.maxz, self.minz = torch.amax(self._xyz[:,2]), torch.amin(self._xyz[:,2]) 
        self.maxy, self.miny = torch.amax(self._xyz[:,1]), torch.amin(self._xyz[:,1]) 
        self.maxx, self.minx = torch.amax(self._xyz[:,0]), torch.amin(self._xyz[:,0]) 
        self.maxz = min((self.maxz, 200.0)) # some outliers in the n4d datasets.. 
        if self.args.dsh:
            self.sh_mlp.to("cuda")
        self.init_key_time()
        # def add_mlp(mlp,mlp_name,weight_name=''):
        #     for name, W in self.mlp.named_parameters():
        #         if weight_name in name:
        #             self.mlp_grd[mlp_name+name] = torch.zeros_like(W, requires_grad=False).cuda()

        # add_mlp(self.motion_mlp,"motion")
        # add_mlp(self.rot_mlp,"rot")
        # add_mlp(self.opacity_mlp,"opacity")
        # add_mlp(self.hexplane,"hexplane","grids")
        # for name, W in self.motion_mlp.named_parameters():
        #     self.mlp_grd["motion"+name] = torch.zeros_like(W, requires_grad=False).cuda()
        # for name, W in self.rot_mlp.named_parameters():
        #     self.mlp_grd["rot"+name] = torch.zeros_like(W, requires_grad=False).cuda()
        # for name, W in self.hexplane.named_parameters():
        #     if "grids" in name:
        #         self.mlp_grd["hexplane"+name] = torch.zeros_like(W, requires_grad=False).cuda()
    def init_key_time(self):
        self.frame_interval = int((self.duration -1) / (self.key_frame_nums-1))
        self.time_interval = self.frame_interval/self.duration
        self.key_time_dict = {}
        for i in range(self.key_frame_nums-1):
            self.key_time_dict[i*self.time_interval] = i
        self.key_time_dict[(self.duration-1)/self.duration] = self.key_frame_nums-1
        self.key_index_dict={v:k for k,v in self.key_time_dict.items()}
        self.key_frame_dict = {}
        for i in range(self.key_frame_nums-1):
            self.key_frame_dict[i] = [j for j in range(self.duration) if j/self.duration >= self.key_index_dict[i] and j/self.duration < self.key_index_dict[i+1]]
        self.key_frame_dict[self.key_frame_nums-2].append(self.duration-1)

        #vali
        frame_num_list = [len(v) for v in self.key_frame_dict.values()]
        print(self.key_frame_dict)
        assert sum(frame_num_list) == self.duration
        # self.key_time_dict[(self.duration-1)/self.duration] = self.key_frame_nums
    def static2dynamatic(self):
        #这里不能用expand,expand不会分配新内存，很危险
        new_xyz = self._xyz.unsqueeze(1).repeat(1,self.key_frame_nums,1)
        new_scaling = self._scaling.unsqueeze(1).repeat(1,self.key_frame_nums,1)
        new_rotation = self._rotation.unsqueeze(1).repeat(1,self.key_frame_nums,1)
        if self.args.dsh:
            new_features_dc = self._features_dc.unsqueeze(1).repeat(1,self.key_frame_nums,1,1)
            new_features_rest = self._features_rest.unsqueeze(1).repeat(1,self.key_frame_nums,1,1)
        else:
            new_features_dc = self._features_dc
            new_features_rest = self._features_rest
        new_opacity = self._opacity.unsqueeze(1).repeat(1,self.key_frame_nums,1)

        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacity,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        }
        optimizable_tensors = {}
        for key,tensor in d.items():
            print(key,tensor.shape)
            optimizable_tensors.update(self.replace_tensor_to_optimizer(tensor,key))
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        # self._trbf_center = optimizable_tensors["trbf_center"]
        # self._trbf_scale = optimizable_tensors["trbf_scale"]
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0],self.key_frame_nums), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda") #所有valid的点都会参与计算

        self.valid_mask = torch.ones((self.get_xyz.shape[0],self.key_frame_nums),device="cuda",dtype=bool) #全为zero了，才让这个点给过滤掉
        self.is_dynamatic = True
    def cache_gradient(self,stage):
        '''把grad都传给grd'''
        if stage == "dynamatic":#只有dynamtic时，才会有下面这三个的梯度
            # self._motion_grd += self._motion.grad.clone()
            # self._motion_fourier_grd += self._motion_fourier.grad.clone()
            # self._omega_grd += self._omega.grad.clone()

            def add_mlp(mlp,mlp_name,weight_name=''):
                for name, W in mlp.named_parameters():
                    if weight_name in name and W.grad is not None:
                        self.mlp_grd[mlp_name+name] += W.grad.clone()
            
            add_mlp(self.motion_mlp,"motion")
            add_mlp(self.rot_mlp,"rot")
            add_mlp(self.opacity_mlp,"opacity")
            # add_mlp(self.forward_field,"forward","grids")
            # add_mlp(self.backward_field,"backward","grids")
            add_mlp(self.deform_field,"deform","grids")
            if self.args.dsh:
                add_mlp(self.sh_mlp,"sh")
            # for name, W in self.motion_mlp.named_parameters():
            #     # if 'weight' in name:
            #         self.mlp_grd["motion"+name] = self.mlp_grd["motion"+name] + W.grad.clone()
            # for name, W in self.rot_mlp.named_parameters():
            #     # if 'weight' in name:
            #         self.mlp_grd["rot"+name] = self.mlp_grd["rot"+name] + W.grad.clone()

            # for name, W in self.hexplane.named_parameters():
            #     if "grids" in name:
            #         # print(name)
            #         self.mlp_grd["hexplane"+name] = self.mlp_grd["hexplane"+name] + W.grad.clone()
        # else:
        #     self._xyz_grd += self._xyz.grad.clone()
        #     self._rotation_grd += self._rotation.grad.clone()


        self._xyz_grd += self._xyz.grad.clone()
        self._features_dc_grd += self._features_dc.grad.clone()
        self._scaling_grd += self._scaling.grad.clone()
        self._rotation_grd += self._rotation.grad.clone()
        self._opacity_grd += self._opacity.grad.clone()
        # print(self._opacity_grd)
        self._features_rest_grd += self._features_rest.grad.clone()

        # self._hexplane_grd += self.hexplane.grad.clone()
        # self._trbf_center_grd += self._trbf_center.grad.clone()
        # self._trbf_scale_grd += self._trbf_scale.grad.clone()
        
        

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
        if self.is_dynamatic and self.args.time_batch:
            self.time_batch = torch.zeros_like(self._opacity, requires_grad=False,dtype=torch.uint8)
        else:
            self.time_batch = None
        # self._trbf_center_grd = torch.zeros_like(self._trbf_center, requires_grad=False)
        # self._trbf_scale_grd = torch.zeros_like(self._trbf_scale, requires_grad=False)
        # self._motion_grd = torch.zeros_like(self._motion, requires_grad=False)
        # self._motion_fourier_grd = torch.zeros_like(self._motion_fourier, requires_grad=False)

        # self._omega_grd = torch.zeros_like(self._omega, requires_grad=False)
        # self.end_grd = torch.zeros_like(self.end, requires_grad=False)
        # self._hexplane_grd = torch.zeros_like(self.hexplane.grids, requires_grad=False)
        # self.motion_mlp.grd = torch.zeros_like(self.motion_mlp, requires_grad=False)
        # self.rot_mlp.grd = torch.zeros_like(self.rot_mlp, requires_grad=False)

        # self._color_change_grd = torch.zeros_like(self._color_change, requires_grad=False)
        # self.timescale_grd = torch.zeros_like(self.timescale, requires_grad=False)


        for name in self.mlp_grd.keys():
            self.mlp_grd[name].zero_()

    def set_batch_gradient(self, cnt,stage):
        ratio = 1/cnt
        if self.time_batch is not None:
            nonzero_indices = torch.nonzero(self.time_batch, as_tuple=False)
            if not self.args.dsh:
                self._features_dc.grad = self._features_dc_grd * ratio
                self._features_rest.grad = self._features_rest_grd * ratio
            else:
                raise NotImplementedError
            # 将非零元素的倒数写入结果张量中
            # self._features_dc_grd[nonzero_indices[:, 0], nonzero_indices[:, 1]] = self._features_dc_grd[nonzero_indices[:, 0], nonzero_indices[:, 1]]*1.0 / self.time_batch[nonzero_indices[:, 0], nonzero_indices[:, 1]]
            # self._features_rest_grd[nonzero_indices[:, 0], nonzero_indices[:, 1]] = self._features_rest_grd[nonzero_indices[:, 0], nonzero_indices[:, 1]]*1.0 / self.time_batch[nonzero_indices[:, 0], nonzero_indices[:, 1]]
            # print(nonzero_indices.shape,self.time_batch.shape)
            # print( self.time_batch[nonzero_indices[:, 0], nonzero_indices[:, 1]])
            # print(self._xyz_grd[nonzero_indices[:, 0],nonzero_indices[:, 1]].shape)
            # print((1.0 / self.time_batch[nonzero_indices[:, 0], nonzero_indices[:, 1]]).shape)
            # print(self._opacity_grd)
            self._xyz_grd[nonzero_indices[:, 0],nonzero_indices[:, 1]] = self._xyz_grd[nonzero_indices[:, 0],nonzero_indices[:, 1]]*(1.0 / self.time_batch[nonzero_indices[:, 0], nonzero_indices[:, 1]])
            self._scaling_grd[nonzero_indices[:, 0],nonzero_indices[:, 1]] = self._scaling_grd[nonzero_indices[:, 0],nonzero_indices[:, 1]]*1.0 / self.time_batch[nonzero_indices[:, 0], nonzero_indices[:, 1]]
            self._rotation_grd[nonzero_indices[:, 0],nonzero_indices[:, 1]] = self._rotation_grd[nonzero_indices[:, 0],nonzero_indices[:, 1]]*1.0 / self.time_batch[nonzero_indices[:, 0], nonzero_indices[:, 1]]
            self._opacity_grd[nonzero_indices[:, 0],nonzero_indices[:, 1]] = self._opacity_grd[nonzero_indices[:, 0],nonzero_indices[:, 1]]*1.0 / self.time_batch[nonzero_indices[:, 0], nonzero_indices[:, 1]]
            self._features_dc_grad = self._features_dc_grd
            self._features_rest_grad = self._features_rest_grd
            self._xyz_grad = self._xyz_grd
            self._scaling_grad = self._scaling_grd
            self._rotation_grad = self._rotation_grd
            self._opacity_grad = self._opacity_grd
            # print(self._opacity_grad)
        else:
            '''把grd通过batch平均后传给grad'''

            self._features_dc.grad = self._features_dc_grd * ratio
            self._features_rest.grad = self._features_rest_grd * ratio
            self._xyz.grad = self._xyz_grd * ratio
            self._scaling.grad = self._scaling_grd * ratio
            self._rotation.grad = self._rotation_grd * ratio
            self._opacity.grad = self._opacity_grd * ratio
        # self._trbf_center.grad = self._trbf_center_grd * ratio
        # self._trbf_scale.grad = self._trbf_scale_grd* ratio
        # self._motion.grad = self._motion_grd * ratio
        # self._motion_fourier.grad = self._motion_fourier_grd * ratio

        # self._omega.grad = self._omega_grd * ratio


        # self.hexplane.grad = self._hexplane_grd * ratio
        # self.motion_mlp.grad = self.motion_mlp.grd * ratio
        # self.rot_mlp.grad = self.rot_mlp.grd * ratio
        # self._color_change.grad = self._color_change_grd * ratio
        # self.timescale.grad = self.timescale_grd * ratio
        # print("motion_mlp",self.motion_mlp.parameters())
        if stage == "dynamatic":

            def set_mlp_gradient(mlp,mlp_name,weight_name=''):
                for name, W in mlp.named_parameters():                    
                    if weight_name in name :
                        W.grad = self.mlp_grd[mlp_name+name]*ratio

            set_mlp_gradient(self.motion_mlp,"motion")
            set_mlp_gradient(self.rot_mlp,"rot")
            set_mlp_gradient(self.opacity_mlp,"opacity")
            # set_mlp_gradient(self.forward_field,"forward","grids")
            # set_mlp_gradient(self.backward_field,"backward","grids")
            set_mlp_gradient(self.deform_field,"deform","grids")
            if self.args.dsh:
                set_mlp_gradient(self.sh_mlp,"sh")
            # for name, W in self.motion_mlp.named_parameters():
            #     # if 'weight' in name:
            #         # print(name,W)
            #         W.grad = self.mlp_grd["motion"+name] * ratio
            # for name, W in self.rot_mlp.named_parameters():
            #     # if 'weight' in name:
            #         W.grad = self.mlp_grd["rot"+name] * ratio

            # for name, W in self.hexplane.named_parameters():
            #     if "grids" in name:
            #         W.grad = self.mlp_grd["hexplane"+name] * ratio

    def training_setup(self, training_args):
        '''设置optimizer'''
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if training_args.use_weight_decay:
            l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            # {'params': [self._omega], 'lr': training_args.deform_feature_lr,"weight_decay":8e-7,"name": "omega"},
            # {'params': [self._motion], 'lr':   training_args.deform_feature_lr*self.spatial_lr_scale , "weight_decay":8e-7,"name": "motion"},
            {'params': list(self.motion_mlp.parameters()), 'lr':   training_args.mlp_lr , "weight_decay":8e-7,"name": "motion_mlp"},
            {'params': list(self.rot_mlp.parameters()), 'lr':   training_args.mlp_lr , "weight_decay":8e-7,"name": "rot_mlp"},
            {'params': list(self.opacity_mlp.parameters()), 'lr':   training_args.mlp_lr , "weight_decay":8e-7,"name": "opacity_mlp"},
            {'params': list(self.deform_field.parameters()), 'lr':   training_args.hexplane_lr*self.spatial_lr_scale , "weight_decay":8e-7,"name": "deform"}
            # {'params': list(self.forward_field.parameters()), 'lr':   training_args.hexplane_lr*self.spatial_lr_scale , "weight_decay":8e-7,"name": "forward"},
            # {'params': list(self.backward_field.parameters()), 'lr':   training_args.hexplane_lr*self.spatial_lr_scale , "weight_decay":8e-7,"name": "backward"}

                
        ]
            if self.args.dsh:
                l.append({'params': list(self.sh_mlp.parameters()), 'lr':   training_args.mlp_lr , "weight_decay":8e-7,"name": "sh"})
        else:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                # {'params': [self._omega], 'lr': training_args.deform_feature_lr,"name": "omega"},
                # {'params': [self._motion], 'lr':   training_args.deform_feature_lr , "name": "motion"},
                {'params': list(self.motion_mlp.parameters()), 'lr':   training_args.mlp_lr , "name": "motion_mlp"},
                {'params': list(self.rot_mlp.parameters()), 'lr':   training_args.mlp_lr , "name": "rot_mlp"},
                {'params': list(self.deform_field.parameters()), 'lr':   training_args.hexplane_lr*self.spatial_lr_scale ,"name": "deform"},

                # {'params': list(self.forward_field.parameters()), 'lr':   training_args.hexplane_lr*self.spatial_lr_scale ,"name": "forward"},
                {'params': list(self.opacity_mlp.parameters()), 'lr':   training_args.mlp_lr , "name": "opacity_mlp"},
                # {'params': list(self.backward_field.parameters()), 'lr':   training_args.hexplane_lr*self.spatial_lr_scale , "name": "backward"}


                    
            ]
            if self.args.dsh:
                l.append({'params': list(self.sh_mlp.parameters()), 'lr':   training_args.mlp_lr , "weight_decay":8e-7,"name": "sh"})


        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # print(len(self.optimizer.param_groups))
        # print(len(self.optimizer.state))
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_lr,
                                                    lr_final=training_args.mlp_lr_final,
                                                    # lr_delay_mult=training_args.mlp_lr_delay_mult,
                                                    start_step = training_args.static_iteration,
                                                    max_steps=training_args.position_lr_max_steps)
        self.deform_feature_scheduler_args = get_expon_lr_func(lr_init=training_args.deform_feature_lr,
                                                    lr_final=training_args.deform_feature_lr_final,
                                                    # lr_delay_mult=training_args.deform_feature_lr_delay_mult,
                                                    start_step = training_args.static_iteration,
                                                    max_steps=training_args.position_lr_max_steps)
        self.hexplane_scheduler_args = get_expon_lr_func(lr_init=training_args.hexplane_lr,
                                                    lr_final=training_args.hexplane_lr_final,
                                                    # lr_delay_mult=training_args.hexplane_lr_delay_mult,
                                                    start_step = training_args.static_iteration,
                                                    max_steps=training_args.position_lr_max_steps)  
        print("move decoder to cuda")
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
    
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # print("xyz",lr)
                # return lr
            elif "mlp" in param_group["name"] :
                lr = self.mlp_scheduler_args(iteration)
                param_group['lr'] = lr
                # print("mlp",lr)
            elif param_group["name"] == "motion" or param_group["name"] =="omega":
                lr = self.deform_feature_scheduler_args(iteration)
                # lr=0
                param_group['lr'] = lr
            elif param_group["name"] == "deform":
            # elif param_group["name"] == "forward" or param_group["name"] == "backward":
                lr = self.hexplane_scheduler_args(iteration)
                param_group['lr'] = lr
                # print("hexplane",lr)

        # for param_group in self.optimizer.param_groups:
        #     print(param_group['name'],param_group['lr'])
    def construct_list_of_attributes(self):
        # l = ['x', 'y', 'z','trbf_center', 'trbf_scale' ,'nx', 'ny', 'nz'] # 'trbf_center', 'trbf_scale' 
        l = ['x', 'y', 'z' ,'nx', 'ny', 'nz']
        # All channels except the 3 DC

        # for i in range(self._motion.shape[1]):
        #     l.append('motion_{}'.format(i))
        
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
        # for i in range(self._omega.shape[1]):
        #     l.append('omega_{}'.format(i))

        return l

    def construct_list_of_dynamatic_attributes(self):
        #增加k个关键帧
        l=[]

        # All channels except the 3 DC
        for k in range(self.key_frame_nums):
            for i in range(3):
                l.append('xyz_{}_{}'.format(i,k))
        l = l+['nx', 'ny', 'nz']
        if self.args.dsh:
            for k in range(self.key_frame_nums):
                for i in range(self._features_dc.shape[2]*self._features_dc.shape[3]):
                    l.append('f_dc_{}_{}'.format(i,k))
            for k in range(self.key_frame_nums):
                for i in range(self._features_rest.shape[2]*self._features_rest.shape[3]):
                    l.append('f_rest_{}_{}'.format(i,k))
        else:
            for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
                l.append('f_dc_{}'.format(i))
            for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))

        for k in range(self.key_frame_nums):
            l.append('opacity_{}'.format(k))
        for k in range(self.key_frame_nums):
            for i in range(self._scaling.shape[-1]):
                l.append('scale_{}_{}'.format(i,k))
        for k in range(self.key_frame_nums):
            for i in range(self._rotation.shape[-1]):
                l.append('rot_{}_{}'.format(i,k))
        for k in range(self.key_frame_nums):
            l.append('valid_{}'.format(k))


        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        if self.is_dynamatic:
            normals = np.zeros((self._xyz.shape[0],3))
            xyz = self._xyz.detach().transpose(1,2).flatten(start_dim=1).contiguous().cpu().numpy()
            if self.args.dsh:
                f_dc = self._features_dc.detach().permute(0,3,2,1).flatten(start_dim=1).contiguous().cpu().numpy()
                f_rest = self._features_rest.detach().permute(0,3,2,1).flatten(start_dim=1).contiguous().cpu().numpy()
            else:
                f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
                f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()                
            opacities = self._opacity.transpose(1,2).flatten(start_dim=1).contiguous().cpu().numpy()
            scale = self._scaling.transpose(1,2).flatten(start_dim=1).contiguous().cpu().numpy()
            rotation = self._rotation.transpose(1,2).flatten(start_dim=1).contiguous().cpu().numpy()
            valid_mask = self.valid_mask.detach().cpu().numpy()
            dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_dynamatic_attributes()]
            path = path.replace(".ply", "_dy.ply")
            attributes = np.concatenate((xyz, normals, f_dc,f_rest, opacities, scale, rotation,valid_mask), axis=1)
            # print(self._xyz,self._features_dc,self._features_rest,self._opacity,self._scaling,self._rotation)

        else:
            xyz = self._xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            scale = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()
            attributes = np.concatenate((xyz, normals, f_dc,f_rest, opacities, scale, rotation), axis=1)


            # trbf_center= self._trbf_center.detach().cpu().numpy()

            # trbf_scale = self._trbf_scale.detach().cpu().numpy()
            # motion = self._motion.detach().cpu().numpy()
            # motion_fourier = self._motion_fourier.detach().cpu().numpy()

            # omega = self._omega.detach().cpu().numpy()


            dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, trbf_center, trbf_scale, normals, motion, f_dc, opacities, scale, rotation, omega), axis=1)
        # attributes = np.concatenate((xyz, normals, f_dc,f_rest, opacities, scale, rotation), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        model_fname = path.replace(".ply", ".pth")
        print(f'Saving model checkpoint to: {model_fname}')
        # torch.save(self.rgbdecoder.state_dict(), model_fname)
        mlp_dict = {'motion_state_dict': self.motion_mlp.state_dict(), 'rot_state_dict': self.rot_mlp.state_dict(),
        # 'forward_state_dict':self.forward_field.state_dict(),
        "opacity_state_dict":self.opacity_mlp.state_dict(),
        'deform_state_dict':self.deform_field.state_dict()}
        if self.args.dsh:
            mlp_dict['sh_state_dict'] = self.sh_mlp.state_dict()
        torch.save(mlp_dict, model_fname)



    def reset_opacity(self):
        # print(self.get_opacity[self.valid_mask])
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        # print(opacities_new)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        for name,W in self.opacity_mlp.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(W, 1e-3)
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(W)
        # print(self.get_opacity[self.valid_mask])


   
    def load_dy_ply(self,path):
        path = path.replace(".ply", "_dy.ply")
        if not os.path.exists(path):
            raise NotImplementedError
        plydata = PlyData.read(path)
        xyz_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("xyz_")]
        xyz = np.zeros((len(plydata.elements[0]["xyz_0_0"]),len(xyz_names)))
        for idx,attr_name in enumerate(xyz_names):
            xyz[:,idx] = plydata.elements[0][attr_name]
        xyz = xyz.reshape((xyz.shape[0],3,self.key_frame_nums))
        
        opacity_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("opacity_")]
        opacities = np.zeros((xyz.shape[0],len(opacity_names)))
        for idx,attr_name in enumerate(opacity_names):
            opacities[:,idx] = plydata.elements[0][attr_name]
        opacities = opacities.reshape((opacities.shape[0],1,self.key_frame_nums))

        valid_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("valid_")]
        valid = np.zeros((xyz.shape[0],len(valid_names)))
        for idx,attr_name in enumerate(valid_names):
            valid[:,idx] = plydata.elements[0][attr_name]
        valid = valid.reshape((valid.shape[0],self.key_frame_nums))

        rots_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rots_names)))
        for idx, attr_name in enumerate(rots_names):
            rots[:,idx] = plydata.elements[0][attr_name]
        rots = rots.reshape((rots.shape[0], 4, -1))

        scales_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0],  len(scales_names)))
        for idx, attr_name in enumerate(scales_names):
            scales[:, idx] = plydata.elements[0][attr_name]
        scales = scales.reshape((scales.shape[0], 3, -1))

        fdc_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_dc_")]
        f_dc = np.zeros((xyz.shape[0], len(fdc_names)))
        for idx, attr_name in enumerate(fdc_names):
            f_dc[:, idx] = plydata.elements[0][attr_name]
        fextra_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        f_extra = np.zeros((xyz.shape[0], len(fextra_names)))
        for idx, attr_name in enumerate(fextra_names):
            f_extra[:, idx] = plydata.elements[0][attr_name]        
        
        if self.args.dsh:
            f_dc = f_dc.reshape((f_dc.shape[0], 3,1, -1))
            f_extra = f_extra.reshape((f_extra.shape[0], 3,(self.max_sh_degree + 1) ** 2 - 1, -1))
            self._features_dc = nn.Parameter(torch.tensor(f_dc, dtype=torch.float, device="cuda").permute(0,3,2,1).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(torch.tensor(f_extra, dtype=torch.float, device="cuda").permute(0,3,2,1).contiguous().requires_grad_(True))
        else:
            f_dc = f_dc.reshape((f_dc.shape[0], 3, -1))
            f_extra = f_extra.reshape((f_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
            self._features_dc = nn.Parameter(torch.tensor(f_dc, dtype=torch.float, device="cuda").permute(0,2,1).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(torch.tensor(f_extra, dtype=torch.float, device="cuda").permute(0,2,1).contiguous().requires_grad_(True))
        
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").permute(0,2,1).contiguous().requires_grad_(True))
        # print(self._xyz)
        # self._features_dc = nn.Parameter(torch.tensor(f_dc, dtype=torch.float, device="cuda").permute(0,3,2,1).contiguous().requires_grad_(True))
        # print(self._features_dc)
        # self._features_rest = nn.Parameter(torch.tensor(f_extra, dtype=torch.float, device="cuda").permute(0,3,2,1).contiguous().requires_grad_(True))
        # print(self._features_rest)
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").permute(0,2,1).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").permute(0,2,1).contiguous().requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").permute(0,2,1).contiguous().requires_grad_(True))
        # self._motion = nn.Parameter(torch.tensor(motion, dtype=torch.float, device="cuda").requires_grad_(True))
        # print(self._xyz,self._features_dc,self._features_rest,self._opacity,self._scaling,self._rotation)

        # self._omega = nn.Parameter(torch.tensor(omegas, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        self.computedopacity =self.opacity_activation(self._opacity)
        self.computedscales = torch.exp(self._scaling) # change not very large
        self.valid_mask = torch.tensor(valid,dtype=bool,device="cuda")
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0],self.key_frame_nums), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.load_mlp(path)
        self.init_key_time()
        self.is_dynamatic = True
    def load_ply(self, path):
        if not os.path.exists(path):
            self.load_dy_ply(path)
            return
        plydata = PlyData.read(path)


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


        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._motion = nn.Parameter(torch.tensor(motion, dtype=torch.float, device="cuda").requires_grad_(True))

        # self._omega = nn.Parameter(torch.tensor(omegas, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        self.computedopacity =self.opacity_activation(self._opacity)
        self.computedscales = torch.exp(self._scaling) # change not very large

        self.load_mlp(path)
        self.init_key_time()
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # for name, W in self.motion_mlp.named_parameters():
        #     self.mlp_grd["motion"+name] = torch.zeros_like(W, requires_grad=False).cuda()
        # for name, W in self.rot_mlp.named_parameters():
        #     self.mlp_grd["rot"+name] = torch.zeros_like(W, requires_grad=False).cuda()
        # for name, W in self.hexplane.named_parameters():
        #     if "grids" in name:
        #         self.mlp_grd["hexplane"+name] = torch.zeros_like(W, requires_grad=False).cuda()

    def load_mlp(self,path):
        mlp_state_dict = torch.load(path.replace(".ply", ".pth"))
        self.motion_mlp.load_state_dict({k.replace('motion_', ''): v for k, v in mlp_state_dict['motion_state_dict'].items()})
        self.rot_mlp.load_state_dict({k.replace('rot_', ''): v for k, v in mlp_state_dict['rot_state_dict'].items()})
        # self.forward_field.load_state_dict({k.replace('forward_', ''): v for k, v in mlp_state_dict['forward_state_dict'].items()})
        # self.backward_field.load_state_dict({k.replace('backward_', ''): v for k, v in mlp_state_dict['backward_state_dict'].items()})
        self.deform_field.load_state_dict({k.replace('deform_', ''): v for k, v in mlp_state_dict['deform_state_dict'].items()})

        self.opacity_mlp.load_state_dict({k.replace('opacity_', ''): v for k, v in mlp_state_dict['opacity_state_dict'].items()})
        self.motion_mlp.to("cuda")
        self.rot_mlp.to("cuda")
        self.deform_field.to("cuda")
        # self.forward_field.to("cuda")
        # self.backward_field.to("cuda")
        self.opacity_mlp.to("cuda")
        print(self.args.dynamatic_mlp)
        if self.args.dsh:
            self.sh_mlp.load_state_dict({k.replace('sh_', ''): v for k, v in mlp_state_dict['sh_state_dict'].items()})
            self.sh_mlp.to("cuda")
        self.mlp_grd = {}
        self.init_mlp_grd()
    def init_mlp_grd(self):
        def add_mlp(mlp,mlp_name,weight_name=''):
            for name, W in mlp.named_parameters():
                if weight_name in name:
                    self.mlp_grd[mlp_name+name] = torch.zeros_like(W, requires_grad=False).cuda()

        add_mlp(self.motion_mlp,"motion")
        add_mlp(self.rot_mlp,"rot")
        add_mlp(self.opacity_mlp,"opacity")
        add_mlp(self.deform_field,"deform","grids")
        # add_mlp(self.forward_field,"forward","grids")
        # add_mlp(self.backward_field,"backward","grids")
        if self.args.dsh:
            add_mlp(self.sh_mlp,"sh")
    def replace_tensor_to_optimizer(self, tensor, name):
        '''将optim中对应name的值给换成tensor，并且adam中原本保存的状态清0'''
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)

                if stored_state is not None:
                    stored_state = self.optimizer.state.get(group['params'][0], None)
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # print(group,len(group["params"]))
            # print(group["name"],group["params"][0].shape)
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
        # if self.valid_mask is not None:

        #     valid_mask = ~mask
        #     self.valid_mask = torch.logical_and(valid_mask,self.valid_mask)

        #     #将左右两边为false的点去掉
        #     right_shift=torch.cat((torch.zeros((self.valid_mask.shape[0],1),device="cuda",dtype=bool),self.valid_mask[:,:-1]),dim=1)
        #     left_shift=torch.cat((self.valid_mask[:,1:],torch.zeros((self.valid_mask.shape[0],1),device="cuda",dtype=bool)),dim=1)
        #     self.valid_mask = torch.logical_and(self.valid_mask,torch.logical_or(right_shift,left_shift)) #左右两边有一个为true，就将这个点保留
        #     mask = torch.all(~self.valid_mask,dim=1) #如果全为false，则为true #[N]
        
        # print(prune_mask.shape)
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
        # self._motion = optimizable_tensors["motion"]
        # self._omega = optimizable_tensors["omega"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        # print(self.max_radii2D.device)
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        # print(self.max_radii2D.shape)
        # self._motion_fourier = optimizable_tensors["motion_fourier"]
        # self.timescale = optimizable_tensors["timescale"]
        if self.omegamask is not None :
            self.omegamask = self.omegamask[valid_points_mask]
        if self.valid_mask is not None:
            self.valid_mask=self.valid_mask[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        '''将tensors cat 到optimizer中，即加入进去'''
        optimizable_tensors = {}
        # print(len(self.optimizer.param_groups))
        # print(len(self.optimizer.state))
        for group in self.optimizer.param_groups:
            if len(group["params"]) == 1 and group["name"] in tensors_dict:
                # print(group["name"],tensors_dict[group["name"]].shape)

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

    def densification_postfix(self, new_xyz, new_features_dc, new_feature_rest ,new_opacities, new_scaling, new_rotation, dummy=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_feature_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        # "trbf_center" : new_trbf_center,
        # "trbf_scale" : new_trbfscale,
        # "motion": new_motion,
        # "omega": new_omega
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d) #将这些点加入进去
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        # self._trbf_center = optimizable_tensors["trbf_center"]
        # self._trbf_scale = optimizable_tensors["trbf_scale"]
        # self._motion = optimizable_tensors["motion"]
        # self._omega = optimizable_tensors["omega"]
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # if self.is_dynamatic:
        #     self.max_radii2D = torch.zeros((self.get_xyz.shape[0],self.key_frame_nums), device="cuda")
        # else:
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    

    def densify_and_splitv2(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # print(selected_pts_mask,selected_pts_mask.sum(),selected_pts_mask.shape)
        # print(self.is_dynamatic)
        if self.is_dynamatic:
            # selected_pts_mask = torch.logical_and(selected_pts_mask,
            #                                   torch.max(torch.max(self.get_scaling, dim=2).values[self.valid_mask],dim=1) > self.percent_dense*scene_extent)#在valid的点中，最大的scale满足条件的情况下才去克隆
            
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(torch.max(self.get_scaling, dim=2).values.masked_fill_(~self.valid_mask,0),dim=1).values > self.percent_dense*scene_extent)#在valid的点中，最大的scale满足条件就可以去split
            # print(selected_pts_mask)
            selected_scale_mask = torch.where(torch.max(self.get_scaling, dim=2).values[selected_pts_mask] > self.percent_dense*scene_extent,True,False)
            # print(selected_scale_mask)
            # print(self.get_scaling[selected_pts_mask])
            # print(self.get_scaling[selected_pts_mask].shape)
            # print(selected_scale_mask.shape)
            new_scaling = self.get_scaling[selected_pts_mask]
            new_scaling[selected_scale_mask] /= (0.8*N)
            new_scaling = self.scaling_inverse_activation(new_scaling.repeat(N,1,1))
            # print(new_scaling[0],new_scaling[1])
            stds = self.get_scaling[selected_pts_mask][selected_scale_mask] #[B,3]
            # print("stds",stds.shape,stds)
            means =torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            # print("samples",samples.shape)
            rots = build_rotation(self._rotation[selected_pts_mask][selected_scale_mask])#[B,3,3]
            # print("rots",rots.shape)
            new_xyz = self.get_xyz[selected_pts_mask] #[B,8,3]
            # print(new_xyz[0],new_xyz[1],new_xyz.shape)
            new_xyz[selected_scale_mask] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            # print(new_xyz[0],new_xyz[1],new_xyz.shape)
            new_xyz = new_xyz.repeat(N,1,1)
            # print(new_xyz[0],new_xyz[1],new_xyz.shape)
            # exit(0)
            new_rotation = self._rotation[selected_pts_mask].repeat(N,1,1)
            if self.args.dsh:
                new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1,1) # n,1,1 to n1
                new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1,1)
            else:
                new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1) # n,1,1 to n1
                new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)             
            new_opacity = self._opacity[selected_pts_mask].repeat(N,1,1)
            
        else:
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))

        # selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                       torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        #trbfmask =  self.trbfoutput > 0.8
        #selected_pts_mask = torch.logical_and(selected_pts_mask, trbfmask.squeeze(1))
            stds = self.get_scaling[selected_pts_mask].repeat(N,1)
            means =torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
            new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1) # n,1,1 to n1
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
            new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        # new_trbf_center = self._trbf_center[selected_pts_mask].repeat(N,1)
        # new_trbf_center = torch.rand_like(new_trbf_center) #* 0.5
        # new_trbf_scale = self._trbf_scale[selected_pts_mask].repeat(N,1)
        # new_motion = self._motion[selected_pts_mask].repeat(N,1)
        # new_omega = self._omega[selected_pts_mask].repeat(N,1)

        # self.densification_postfix(new_xyz, new_features_dc, new_opacity, new_scaling, new_rotation, new_trbf_center, new_trbf_scale, new_motion, new_omega)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,new_opacity, new_scaling, new_rotation)
        if self.valid_mask is not None:
            # print(self.valid_mask.shape)
            new_valid_mask = self.valid_mask[selected_pts_mask].repeat(N,1)
            self.valid_mask = torch.cat((self.valid_mask,new_valid_mask),dim=0)
            # print(self.valid_mask.shape)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(),device="cuda",dtype=bool)),dim=0)
        self.prune_points(prune_filter)
        if self.valid_mask is not None:

            assert self.valid_mask.shape[0] == self._xyz.shape[0]




    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # print(selected_pts_mask,selected_pts_mask.sum(),selected_pts_mask.shape)
        if self.is_dynamatic:
            # print(torch.max(self.get_scaling, dim=2).values.shape)
            # print(torch.max(self.get_scaling, dim=2).values.masked_fill_(~self.valid_mask,0).shape)
            # print(self.valid_mask)
            # print(torch.max(self.get_scaling, dim=2).values.masked_fill_(~self.valid_mask,0))
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(torch.max(self.get_scaling, dim=2).values.masked_fill_(~self.valid_mask,0),dim=1).values <= self.percent_dense*scene_extent)#在valid的点中，最大的scale满足条件的情况下才去克隆
        else:
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
        # new_motion = self._motion[selected_pts_mask]

        # new_omega = self._omega[selected_pts_mask]
        #self.trbfoutput = torch.cat((self.trbfoutput, torch.zeros(N , 1).to(self.trbfoutput)))
        # self.densification_postfix(new_xyz, new_features_dc, new_opacities, new_scaling, new_rotation, new_trbf_center, new_trbfscale, new_motion, new_omega)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,new_opacities, new_scaling, new_rotation)
        
        if self.valid_mask is not None:
            new_valid_mask = self.valid_mask[selected_pts_mask]
            self.valid_mask = torch.cat((self.valid_mask,new_valid_mask),dim=0)
            assert self.valid_mask.shape[0] == self._xyz.shape[0]
            # print(self.valid_mask.shape)



    def densify_pruneclone(self, max_grad, min_opacity, extent, max_screen_size, splitN=1):
        #print("before", torch.amax(self.get_scaling))
        grads = self.xyz_gradient_accum / self.denom
        print("gradient",self.xyz_gradient_accum,grads)

        grads[grads.isnan()] = 0.0
        with torch.no_grad():
            print(torch.mean(grads), torch.amax(grads), torch.amin(grads))
            print(torch.mean(self.get_opacity), torch.amax(self.get_opacity), torch.amin(self.get_opacity))
        print("befre clone", self._xyz.shape[0])
        self.densify_and_clone(grads, max_grad, extent)
        print("after clone", self._xyz.shape[0])

        self.densify_and_splitv2(grads, max_grad, extent, 2)
        print("after split", self._xyz.shape[0])

        prune_mask = (self.get_opacity < min_opacity).squeeze() #[N,K]
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size  #[N]
            big_points_vs = big_points_vs.unsqueeze(1).repeat(1, self.get_xyz.shape[1]) #[N,K]
            big_points_ws = self.get_scaling.max(dim=2).values > 0.1 * extent #[N,K]

            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        # print(self.valid_mask.shape)
        if self.valid_mask is not None:
            
            valid_mask = ~prune_mask
            self.valid_mask = torch.logical_and(valid_mask,self.valid_mask)

            #将左右两边为false的点去掉
            # right_shift=torch.cat((torch.zeros((self.valid_mask.shape[0],1),device="cuda",dtype=bool),self.valid_mask[:,:-1]),dim=1)
            # left_shift=torch.cat((self.valid_mask[:,1:],torch.zeros((self.valid_mask.shape[0],1),device="cuda",dtype=bool)),dim=1)
            # self.valid_mask = torch.logical_and(self.valid_mask,torch.logical_or(right_shift,left_shift)) #左右两边有一个为true，就将这个点保留
            prune_mask = torch.all(~self.valid_mask,dim=1) #如果全为false，则为true #[N]
        # print(prune_mask.shape)
        self.prune_points(prune_mask)
        print("after prune", self._xyz.shape[0])
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter,valid_mask=None):
        if valid_mask is not None:
            # print("valid_mask",valid_mask.shape,update_filter.shape)
            # print(viewspace_point_tensor.grad)
            combined_mask = torch.zeros_like(valid_mask,dtype=bool,device="cuda")
            combined_mask[valid_mask]=update_filter
            self.xyz_gradient_accum[combined_mask] +=torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)

            # self.xyz_gradient_accum[valid_mask][update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)        
            #直接用这个两个索引的方法无法正确更新self.xyz_gradient_accum的值
            self.denom[combined_mask] += 1
        else:
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
        #给visualize用的
        time_embbed = self.time_emb(torch.tensor([timestamp],dtype=torch.float, device="cuda"))
        # print(time_embbed)
        time_embbed = time_embbed.repeat(self._motion.shape[0],1)
        motion_input = torch.cat((self._motion,time_embbed),dim=1)

        residual = self.motion_mlp(motion_input)
        motion = self._xyz.detach() + residual

        return motion

    def get_motion_l1loss(self):
        return torch.mean(torch.norm(self._motion, p=1,dim=1)) + torch.mean(torch.norm(self._motion_fourier, p=1,dim=1))

    # def adptive_timescale(self,t):
    #     '''return: [N,]'''
    #     return self.timescale@torch.tensor([t,1.],dtype=torch.float, device="cuda",requires_grad=False).unsqueeze(1) #N,2 @ 2,
    
    def get_time_smoothloss(self,timestamp):
        # print(self._ddim)

        delta_t = timestamp + 1/self.duration

        current_motion,current_rot = self.get_deformation(timestamp)
        next_motion,next_rot = self.get_deformation(delta_t)

        current = torch.cat((current_motion,current_rot),dim=1)
        next = torch.cat((next_motion,next_rot),dim=1)
        smoothloss = torch.norm(current - next, p=2,dim=1).mean()
        # print(smoothloss)
        return smoothloss

    def get_rot_l1loss(self):
        return torch.mean(torch.norm(self._omega, p=1,dim=1))
    
        
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

    def get_base_feature(self,timestamp):
        frame_interval = (self.duration -1) / self.key_frame_nums
        time_interval = self.key_frame_nums/self.duration
        key_index = int(timestamp / time_interval)
        forward_mask = self.mask[:,key_index]
        backward_mask = self.mask[:,key_index]
        base_mask = torch.logical_and(forward_mask,backward_mask)
        forward_x,forward_r,forward_scale,forward_opacity = self._xyz[:,key_index][base_mask],self._rotation[:,key_index][base_mask],self._scaling[:,key_index][base_mask],self._opacity[:,key_index][base_mask]
        backward_x,backward
        # is_key=False
        # if (key_index+1)*frame_interval/self.duration == time_stamp:
        #     is_key=True
        return base_x,base_r,base_scale,base_opacity,base_mask
    def get_deformation(self,timestamp):
        # print(timestamp)
        # for t,v in self.key_time_dict.items():
        #     print(t,v)
        # print(self._features_dc.shape,self._features_rest.shape)
        if timestamp in self.key_time_dict.keys():
            key_index = self.key_time_dict[timestamp]
            base_mask = self.valid_mask[:,key_index]
            # print(self._xyz.shape,self._rotation.shape,self._scaling.shape,self._opacity.shape)
            features_dc = self._features_dc[:,0][base_mask]
            features_rest = self._features_rest[:,0][base_mask]
            features = torch.cat((features_dc,features_rest),dim=1)#暂时让他们都用第一个的shs
            # base_x,base_r,base_scale,base_opacity = self._xyz[:,key_index][base_mask],self._rotation[:,key_index][base_mask],self._scaling[:,key_index][base_mask],self._opacity[:,key_index][base_mask]
            if self.args.abtest1:
                forward_x,forward_r,forward_scale,forward_opacity,forward_features = self.get_deform(timestamp,key_index,base_mask,0)
                print("backward",key_index-1)
                backward_x,backward_r,backward_scale,backward_opacity,backward_features = self.get_deform(timestamp,key_index-1,base_mask,0,forward=False)
                base_x,base_r,base_scale,base_opacity = (forward_x+backward_x)/2,(forward_r+backward_r)/2,(forward_scale+backward_scale)/2,(forward_opacity+backward_opacity)/2
                features = (forward_features+backward_features)/2

            else:
                if self.time_batch is not None:
                    self.time_batch[base_mask,key_index] += 1
                if key_index ==0:
                    base_x,base_r,base_scale,base_opacity,features = self.get_deform(timestamp,key_index,base_mask,0)
                elif key_index==self.key_frame_nums-1:
                    base_x,base_r,base_scale,base_opacity,features = self.get_deform(timestamp,key_index-1,base_mask,0,forward=False)
                else:
                    forward_x,forward_r,forward_scale,forward_opacity,forward_features = self.get_deform(timestamp,key_index,base_mask,0)
                    print("backward",key_index-1)
                    backward_x,backward_r,backward_scale,backward_opacity,backward_features = self.get_deform(timestamp,key_index-1,base_mask,0,forward=False)
                    base_x,base_r,base_scale,base_opacity = (forward_x+backward_x)/2,(forward_r+backward_r)/2,(forward_scale+backward_scale)/2,(forward_opacity+backward_opacity)/2
                    features = (forward_features+backward_features)/2
            # rot = self.rotation_activation(base_r)
            # scale = self.scaling_activation(base_scale)
            # opacity = self.opacity_activation(base_opacity)
            return base_x,base_r,base_scale,base_opacity,features,base_mask,(key_index,)

        #计算keyindex
        # frame_interval = int((self.duration -1) / (self.key_frame_nums-1))
        # time_interval = frame_interval/self.duration
        for key_time in self.key_time_dict.keys():
            if key_time < timestamp:
                key_index = self.key_time_dict[key_time]
            else:
                break
        # key_index = int(timestamp / self.time_interval)
        time_interval = self.time_interval if key_index != self.key_frame_nums-2 else self.key_index_dict[self.key_frame_nums-1] - self.key_index_dict[self.key_frame_nums-2]
        assert timestamp > self.key_index_dict[key_index] and timestamp < self.key_index_dict[key_index+1] #确保keyindex算的正确

        #计算base_mask,认为前后都有的才是存在的
        forward_mask = self.valid_mask[:,key_index]
        backward_mask = self.valid_mask[:,key_index+1]
        base_mask = torch.logical_and(forward_mask,backward_mask) #[B]
        if self.time_batch is not None:
            # print(key_index)
            self.time_batch[base_mask,key_index] += 1
            self.time_batch[base_mask,key_index+1] += 1
            # print(self.time_batch)
        forward_x,forward_r,forward_scale,forward_opacity,forward_features = self.get_deform(timestamp,key_index,base_mask,time_interval=time_interval)
        # return forward_x,forward_r,forward_scale,forward_opacity,forward_features,base_mask,(key_index,key_index+1)
        backward_x,backward_r,backward_scale,backward_opacity,backward_features = self.get_deform(timestamp,key_index,base_mask,time_interval=time_interval,forward=False)

        # features_dc = self._features_dc[:,0][base_mask]
        # features_rest = self._features_rest[:,0][base_mask]
        # features = torch.cat((features_dc,features_rest),dim=1)#暂时让他们都用第一个的shs

        #combine forward backward
        forward_k=(timestamp- self.key_index_dict[key_index])/time_interval
        assert forward_k>0 and forward_k<1
        motion = (1-forward_k) * forward_x + forward_k*backward_x
        rot = (1-forward_k) * forward_r + forward_k*backward_r
        scale = (1-forward_k) * forward_scale + forward_k*backward_scale
        opacity = (1-forward_k) * forward_opacity + forward_k*backward_opacity
        features = (1-forward_k) * forward_features + forward_k*backward_features
        # motion = forward_k * forward_x + (1-forward_k)*backward_x
        # rot = forward_k * forward_r + (1-forward_k)*backward_r
        # scale = forward_k* forward_scale + (1-forward_k) * backward_scale
        # opacity = forward_k * forward_opacity + (1-forward_k) * backward_opacity

        return motion,rot,scale,opacity,features,base_mask,(key_index,key_index+1)
        # base_x,base_r,base_scale,base_opacity,base_mask = get_base_feature(time_stamp) #得到正向的base_feature

    def get_all_deform(self,time_stamp):
        deform_feature=[]
        time = torch.full((self._xyz.shape[0],1),time_stamp).to("cuda")
        for i in range(self.key_frame_nums):
            deform_feature.append(self.deform_field[i](self._xyz[:,i],time))
        deform_feature = torch.stack(deform_feature,dim=1) #[N,K,D]
        time_feature = torch.tensor([time_stamp - keytime for keytime in self.key_time_dict.keys()],dtype=torch.float, device="cuda")#这里是有正有负的
        time_emb = self.time_emb(time_feature.unsqueeze(1)).unsqueeze(0)
        # print(time_emb.shape)
        time_emb=time_emb.repeat(self._xyz.shape[0],1,1)    #[N,K,D]
        # print(time_emb.shape)
        deform_feature = torch.cat((deform_feature,time_emb),dim=2)
        valid_deform_feature = deform_feature[self.valid_mask] #[N,D]
        motion_residual,rot_scale_residual,opacity_residual = self.motion_mlp(valid_deform_feature),self.rot_mlp(valid_deform_feature),self.opacity_mlp(valid_deform_feature)
        rot_residual = rot_scale_residual[:,:4]
        scale_residual = rot_scale_residual[:,4:]

        #通过time来计算结合的系数
        combine_coef = (1/(abs(time_feature)+1e-4)).unsqueeze(0).repeat(self._xyz.shape[0],1) #[N,K]
        combine_coef[~self.valid_mask] = 0
        #normilize coef
        combine_coef = combine_coef / torch.sum(combine_coef,dim=1).unsqueeze(1)

        motion_feature,rot_feature,scale_feature,opacity_feature =self.get_deform_feature(self._xyz,motion_residual,combine_coef),\
            self.get_deform_feature(self._rotation,rot_residual,combine_coef,self.rotation_activation ),\
                self.get_deform_feature(self._scaling,scale_residual,combine_coef,self.scaling_activation),self.get_deform_feature(self._opacity,opacity_residual,combine_coef,self.opacity_activation)
        sh_features = torch.cat((self._features_dc ,self._features_rest),dim=1)
        return motion_feature,rot_feature,scale_feature,opacity_feature,sh_features            


    def get_deform_feature(self,base_f,res_f,combine_coef,activation=None):
        combine_f = base_f[self.valid_mask] + res_f
        if self.args.pre_activ and activation:
            # print("activ")
            combine_f = activation(combine_f)
        # if activation is not None:
        #     combine_f = activation(combine_f)
        feature = torch.zeros_like(base_f)
        feature[self.valid_mask] = combine_f
        # print("feature",feature)
        # print("coef",combine_coef)
        combine_feature = (feature*(combine_coef.unsqueeze(2))).sum(dim=1)
        if not self.args.pre_activ and activation:
            combine_feature = activation(combine_feature)
        return combine_feature
    def get_deform(self,timestamp,key_index,base_mask,time_interval,forward=True):
        deform_field = self.forward_field if forward else self.backward_field

        key_index = key_index if forward else key_index+1
        # timestamp = timestamp if forward else 1-timestamp - 1/self.duration
        base_x,base_r,base_scale,base_opacity = self._xyz[:,key_index][base_mask],self._rotation[:,key_index][base_mask],self._scaling[:,key_index][base_mask],self._opacity[:,key_index][base_mask]
        # print(base_opacity)
        # print(self.get_opacity[:,key_index][base_mask])
        if self.args.dsh:
            base_f_dc,base_f_rest = self._features_dc[:,key_index][base_mask],self._features_rest[:,key_index][base_mask]
        else:
            base_f_dc = self._features_dc[base_mask]
            base_f_rest = self._features_rest[base_mask]
        time = torch.full((base_x.shape[0],1),timestamp).to("cuda")
        deform_feature = deform_field(base_x,time) #[N,D]
        # print(timestamp,self.key_index_dict[key_index],key_index)
        time_gap = abs(timestamp - self.key_index_dict[key_index])/time_interval if time_interval >0 else 0
        # print(time_gap)
        assert time_gap>=0 and time_gap<1
        time_embbed = self.time_emb(torch.tensor([time_gap],dtype=torch.float, device="cuda"))
        time_embbed = time_embbed.repeat(base_x.shape[0],1)
        deform_feature = torch.cat((deform_feature,time_embbed),dim=1)

        if self.args.dx:
            motion_residual = self.motion_mlp(deform_feature)
            motion = base_x +motion_residual
        else:
            motion = base_x
        
        if self.args.drot:
            rot_residual = self.rot_mlp(deform_feature)
            rot =base_r + rot_residual[:,:4]
            scale = base_scale + rot_residual[:,4:]
            rot = self.rotation_activation(rot)
            scale = self.scaling_activation(scale)
        else:
            rot = self.rotation_activation(base_r)
            scale = self.scaling_activation(base_scale)

        if self.args.dopacity:
            opacity_residual = self.opacity_mlp(deform_feature) #这个的bias要初始化为1

            print(forward,"opacity_residual",opacity_residual)
            print(">0",torch.sum(opacity_residual>0))
            opacity = base_opacity +opacity_residual
            print("o",opacity)
            opacity = self.opacity_activation(opacity)
            print("oo",opacity)
        else:
            opacity = self.opacity_activation(base_opacity)

        if self.args.dsh:
            sh_residual = self.sh_mlp(deform_feature)
            # print(sh_residual.shape)
            features_dc = base_f_dc + sh_residual.unsqueeze(1)
            features_rest = base_f_rest
            features = torch.cat((features_dc,features_rest),dim=1)#暂时让他们都用第一个的shs
            # print(features.shape)
        else:

            features = torch.cat((base_f_dc,base_f_rest),dim=1)#暂时让他们都用第一个的shs

        return motion, rot,scale,opacity,features

    # def get_backward_deform():
    #     pass
    # def get_deformation(self,timestamp):
    #     time_embbed = self.time_emb(torch.tensor([timestamp],dtype=torch.float, device="cuda"))
    #     # print(time_embbed)
    #     time_embbed = time_embbed.repeat(self._motion.shape[0],1)
    #     ####
    #     time = torch.full((self._motion.shape[0],1),timestamp).to("cuda")
    #     hexplane_feature = self.hexplane(self._xyz,time) #[N,D]
    #     deform_feature = torch.cat((hexplane_feature,self._motion,time_embbed),dim=1)

    #     dynamatic = torch.ones_like(self._opacity,device="cuda")

    #     if self.args.dynamatic_mlp:
    #         dynamatic= self.dynamatic_mlp(deform_feature)[:,0].unsqueeze(1) #表示一个点是动态的概率有多大，施加正则让这个值1 or 0
    #         self.dynamatic = dynamatic
    #     if self.args.dx:
    #         motion_residual = self.motion_mlp(deform_feature)
    #         motion = self._xyz + dynamatic*motion_residual
    #     else:
    #         motion = self._xyz
        
    #     if self.args.drot:
    #         rot_residual = self.rot_mlp(deform_feature)
    #         # print("rs",rot_residual)
    #         # print(self._rotation,self._scaling)
    #         rot = self._rotation + dynamatic*rot_residual[:,:4]
    #         scale = self._scaling + dynamatic*rot_residual[:,4:]
    #         rot = self.rotation_activation(rot)
    #         scale = self.scaling_activation(scale)
    #     else:
    #         rot = self.get_rotation
    #         scale = self.get_scaling

    #     if self.args.dopacity:
    #         opacity_residual = self.opacity_mlp(deform_feature) #这个的bias要初始化为1
    #         # print("opacity_residual",opacity_residual)
    #         opacity = self._opacity * dynamatic*opacity_residual
    #         # print("o",opacity)
    #         opacity = self.opacity_activation(opacity)
    #     else:
    #         opacity = self.get_opacity

    #     # if self.args.dshs:

    #     return motion, rot,scale,opacity

    # def get_opacity_deform(self, timestamp):
    #     time_tensor = torch.tensor([timestamp],dtype=torch.float, device="cuda")
    #     end = (time_tensor - self.start_end[:,1]).unsqueeze(1)
    #     start = time_tensor - self.start_end[:,0].unsqueeze(1)
    #     # print(end.shape,start.shape)
    #     opacity = self.get_opacity/ (1.0 + torch.exp(1000.0 * (end)))/ (1 + torch.exp(-1000.0 * (start)))
    #     return opacity
    def set_bounds(self,xyz_max, xyz_min):
        bounds = torch.tensor([xyz_max, xyz_min],dtype=torch.float32,device='cuda')
        self.bounds = nn.Parameter(bounds,requires_grad=False)
        for i in range(self.key_frame_nums):
            self.deform_field[i].set_aabb(xyz_max,xyz_min,self.duration   )
        # self.deform_field.set_aabb(xyz_max,xyz_min,self.duration   )
        # self.backward_field.set_aabb(xyz_max,xyz_min,self.duration)
    def inv_normalize_xyz(self,norm_xyz):
        min_bounds = self.bounds[1]
        max_bounds = self.bounds[0]
        return norm_xyz * (max_bounds - min_bounds) + min_bounds
    def normalize_residual(self,norm_xyz):
        '''0-1 -> -(max_bounds - min_bounds) - (max_bounds - min_bounds)'''
        return (2*norm_xyz-1) * (self.bounds[0] - self.bounds[1])
def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 1,
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