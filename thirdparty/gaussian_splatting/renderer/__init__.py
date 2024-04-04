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

#######################################################################################################################
##### NOTE: CODE IN THIS FILE IS NOT INCLUDED IN THE OVERALL PROJECT'S MIT LICENSE ####################################
##### USE OF THIS CODE FOLLOWS THE COPYRIGHT NOTICE ABOVE #####
#######################################################################################################################




import torch
import math
import time 
import torch.nn.functional as F
import time 




from scene.oursfull import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import getProjectionMatrixCV, focal2fov, fov2focal




def train_ours_full(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale
   

    trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = basicfunction(trbfdistance)
    
    opacity = pointopacity * trbfoutput  # - 0.5
    pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling
    shs = None
    tforpoly = trbfdistanceoffset.detach()
    means3D = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly
    rotations = pc.get_rotation(tforpoly) # to try use 
    colors_precomp = pc.get_features(tforpoly)
    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    rendered_image = pc.rgbdecoder(rendered_image.unsqueeze(0), viewpoint_camera.rays, viewpoint_camera.timestamp) # 1 , 3 #在这里才把9维的feature逐像素的变成RGB color
    rendered_image = rendered_image.squeeze(0)
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth}




def test_ours_full(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    torch.cuda.synchronize()
    startime = time.time()

    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale
   

    trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = basicfunction(trbfdistance)
    
    opacity = pointopacity * trbfoutput  # - 0.5
    pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling
    shs = None
    tforpoly = trbfdistanceoffset.detach()
    means3D = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly
    rotations = pc.get_rotation(tforpoly) # to try use 
    colors_precomp = pc.get_features(tforpoly)
    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    rendered_image = pc.rgbdecoder(rendered_image.unsqueeze(0), viewpoint_camera.rays, viewpoint_camera.timestamp) # 1 , 3
    rendered_image = rendered_image.squeeze(0)
    torch.cuda.synchronize()
    duration = time.time() - startime 

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth,
            "duration":duration}
def test_ours_lite(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None,GRsetting=None, GRzer=None):

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0

    torch.cuda.synchronize()
    startime = time.time()

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    print(pc.bounds)
    print(pc._xyz.max(dim=0),pc._xyz.min(dim=0))
    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False)

    rasterizer = GRzer(raster_settings=raster_settings)
    
    


    tforpoly = viewpoint_camera.timestamp - pc.get_trbfcenter
    rotations = pc.get_rotation(tforpoly) # to try use 
    colors_precomp = pc.get_features(tforpoly)

    
    means2D = screenspace_points #这里有很大的问题，没有去算means3D随时间的变化
   

    cov3D_precomp = None


    shs = None
 
    rendered_image, radii = rasterizer(
        timestamp = viewpoint_camera.timestamp, 
        trbfcenter = pc.get_trbfcenter,
        trbfscale = pc.computedtrbfscale ,
        motion = pc._motion,
        means3D = pc.get_xyz,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = pc.computedopacity,
        scales = pc.computedscales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    
    torch.cuda.synchronize()
    duration = time.time() - startime 
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "duration":duration}




def train_ours_lite(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None,stage=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    这个似乎做不到根据后面的帧去补足前面帧的信息。因为并不是建在一个规范场中的。新增一个点时，只会在xyz空间中做复制。
    理想情况是中间新出现的一个点，既有可能是中间突然出现，也有可能是前面被遮挡到的。要能学习出这个遮挡的信息。
    让一个点被clone之后，跟随原来的点的时间轨迹去移动。刚开始不应该每个时间都有初始化的点，应该让点尽可能是移动过去的。应该来一个让高斯函数的方差尽可能大的约束。

    spg的做法实际上是用了很多高斯来一起拟合原本一个高斯的运动了，这样很像每个帧学一个的策略。缺点是运动可能不一定准确。而且一个点的移动是双向的，不太符合常理
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means.这个东西只是用来记录梯度的，自己的值为0
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    
    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale
   

    trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = basicfunction(trbfdistance)
    
    opacity = pointopacity * trbfoutput  # - 0.5
    pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling

    shs = None
    tforpoly = trbfdistanceoffset.detach()
    # print(means3D[:5],tforpoly,pc._motion[:5])
    means3D = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly
    # print(means3D[:5])
    rotations = pc.get_rotation(tforpoly) # to try use 
    colors_precomp = pc.get_features(tforpoly)

    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth}

def train_ours_flow(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None, stage="dynamatic"):
    """
    Render the scene. 
    
    这个实现不考虑点的opcity随时间改变，仅考虑点随时间的移动，并且点只能向后移动
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means.这个东西只是用来记录梯度的，自己的值为0
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    
    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    # trbfcenter = pc.get_trbfcenter
    # trbfscale = pc.get_trbfscale
   
    # trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes
    # trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    # trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    # trbfoutput = basicfunction(trbfdistance)
    
    opacity = pointopacity   # - 0.5
    # pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling

    shs = None
    # tforpoly = trbfdistanceoffset.detach()
    # print(means3D[:5],tforpoly,pc._motion[:5])
    # means3D = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly
    # print(means3D[:5])
    colors_precomp = pc.get_features(0.0)
    rotations = pc.get_rotation(0.0)
    if stage != "static":
        rotations = pc.get_rotation(viewpoint_camera.timestamp) # to try use 
        colors_precomp = pc.get_features(viewpoint_camera.timestamp)

        means3D = pc.get_motion(viewpoint_camera.timestamp)
    
    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth}



def test_ours_flow(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    torch.cuda.synchronize()
    startime = time.time()

    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    # trbfcenter = pc.get_trbfcenter
    # trbfscale = pc.get_trbfscale
   

    # trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    # trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    # trbfoutput = basicfunction(trbfdistance)
    
    opacity = pointopacity   # - 0.5
    # pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling
    shs = None
    # tforpoly = trbfdistanceoffset.detach()
    rotations = pc.get_rotation(viewpoint_camera.timestamp) # to try use 
    colors_precomp = pc.get_features(viewpoint_camera.timestamp)

    means3D = pc.get_motion(viewpoint_camera.timestamp)

    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # rendered_image = pc.rgbdecoder(rendered_image.unsqueeze(0), viewpoint_camera.rays, viewpoint_camera.timestamp) # 1 , 3
    # rendered_image = rendered_image.squeeze(0)
    torch.cuda.synchronize()
    duration = time.time() - startime 

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth,
            "duration":duration}

def train_ours_flow_full(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None, stage=None):
    """
    Render the scene. 
    基本完全参照gs-flow的实现
    """
    if stage == None:
        raise Exception('没传stage')
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means.这个东西只是用来记录梯度的，自己的值为0
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False
        # ,debug=pipe.debug
        )

    rasterizer = GRzer(raster_settings=raster_settings)

    
    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity


    
    opacity = pointopacity   # - 0.5
    # pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling

    shs = pc.get_features
    colors_precomp = None
    rotations = pc.get_rotation
    # print(scales,rotations)
    # print(pc.timescale.is_leaf,"dd")
    # print(stage)
    if stage != "static":
        means3D,rotations,shs,_ = pc.get_ddim(viewpoint_camera.timestamp)
        # means3D.requires_grad = True
    # print(shs,rotations)
    rendered_image, radii,depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
             "depth": depth
            }

def test_ours_flow_full(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    基本完全参照gs-flow的实现
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means.这个东西只是用来记录梯度的，自己的值为0
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    
    startime = time.time()
    torch.cuda.synchronize()

    # pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False
        # ,debug=pipe.debug
        )

    rasterizer = GRzer(raster_settings=raster_settings)

    
    # means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity


    
    opacity = pointopacity   # - 0.5
    # pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling

    colors_precomp = None
    means3D,rotations,shs,_ = pc.get_ddim(viewpoint_camera.timestamp)
    # print(means3D,pc._xyz,means3D-pc._xyz)
    # means3D = pc._xyz
    rendered_image, radii,depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    torch.cuda.synchronize()
    duration = time.time() - startime

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth,
            "duration": duration
            }

def train_ours_flow_mlp(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None, stage=None):
    """
    Render the scene. 
    基本完全参照gs-flow的实现
    """
    if stage == None:
        raise Exception('没传stage')
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means.这个东西只是用来记录梯度的，自己的值为0
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False
        # ,debug=pipe.debug
        )

    rasterizer = GRzer(raster_settings=raster_settings)

    
    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity


    
    opacity = pointopacity   # - 0.5
    # pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling

    shs = pc.get_features
    colors_precomp = None
    rotations = pc.get_rotation
    # print(scales,rotations)
    # print(pc.timescale.is_leaf,"dd")
    # print(stage)
    if stage == "dynamatic":
        means3D,rotations = pc.get_deformation(viewpoint_camera.timestamp)
        # means3D.requires_grad = True
    # print(shs,rotations)
    # print(means3D,rotations,shs)
    rendered_image, radii,depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
             "depth": depth
            }

def test_ours_flow_mlp(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    基本完全参照gs-flow的实现
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means.这个东西只是用来记录梯度的，自己的值为0
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    
    startime = time.time()
    torch.cuda.synchronize()

    # pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False
        # ,debug=pipe.debug
        )

    rasterizer = GRzer(raster_settings=raster_settings)

    
    # means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity


    
    opacity = pointopacity   # - 0.5
    # pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling
    shs = pc.get_features

    colors_precomp = None
    means3D,rotations = pc.get_deformation(viewpoint_camera.timestamp)
    # print(pc._xyz,means3D)
    # means3D = pc._xyz
    # print(pc._xyz,means3D)
    # rotations = pc.get_rotation
    # means3D = pc._xyz
    rendered_image, radii,depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    torch.cuda.synchronize()
    duration = time.time() - startime

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth,
            "duration": duration
            }

def train_ours_xyz_mlp(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None, stage=None):
    """
    Render the scene. 
    基本完全参照gs-flow的实现
    """
    if stage == None:
        raise Exception('没传stage')
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means.这个东西只是用来记录梯度的，自己的值为0
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False
        # ,debug=pipe.debug
        )

    rasterizer = GRzer(raster_settings=raster_settings)

    
    # means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity


    
    opacity = pointopacity   # - 0.5
    # pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling

    shs = pc.get_features
    colors_precomp = None
    # rotations = pc.get_rotation
    # print(scales,rotations)
    # print(pc.timescale.is_leaf,"dd")
    # print(stage)
    
    means3D,rotations,motion_out = pc.get_deformation(viewpoint_camera.timestamp,stage)
    # print('ddd')
    # print(shs.shape)
    # print(means3D.shape)

        # means3D.requires_grad = True
    # print(shs,rotations)
    rendered_image, radii,depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
             "depth": depth,
             "motion_out": motion_out 
            }

def test_ours_xyz_mlp(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    基本完全参照gs-flow的实现
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means.这个东西只是用来记录梯度的，自己的值为0
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    
    startime = time.time()
    torch.cuda.synchronize()

    # pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False
        # ,debug=pipe.debug
        )

    rasterizer = GRzer(raster_settings=raster_settings)

    
    # means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity


    
    opacity = pointopacity   # - 0.5
    # pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling
    shs = pc.get_features

    colors_precomp = None
    means3D,rotations,_ = pc.get_deformation(viewpoint_camera.timestamp,stage="dynamatic")
    rendered_image, radii,depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    torch.cuda.synchronize()
    duration = time.time() - startime

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth,
            "duration": duration
            }

def train_ours_flow_mlp_opacity(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, GRsetting=None, GRzer=None, stage=None):
    """
    Render the scene. 
    基本完全参照gs-flow的实现
    """
    if stage == None:
        raise Exception('没传stage')
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means.这个东西只是用来记录梯度的，自己的值为0
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False
        # ,debug=pipe.debug
        )

    rasterizer = GRzer(raster_settings=raster_settings)

    
    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity


    
    opacity = pointopacity   # - 0.5
    # pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling

    shs = pc.get_features
    colors_precomp = None
    rotations = pc.get_rotation
    # print(scales,rotations)
    # print(pc.timescale.is_leaf,"dd")
    # print(stage)
    # print(viewpoint_camera.rays.shape)
    if stage == "dynamatic":

        means3D,rotations,scales,opacity,shs,_ = pc.get_deformation(viewpoint_camera.timestamp)
        
        # opacity = pc.get_opacity_deform(viewpoint_camera.timestamp)
        # means3D.requires_grad = True
    # print(means3D.shape,rotations.shape,scales.shape,opacity.shape,means2D.shape,shs.shape)
    # # print(means3D,opacity,scales)
    # print(means3D.isnan().sum(),rotations.isnan().sum(),scales.isnan().sum(),opacity.isnan().sum(),shs.isnan().sum(),means2D.isnan().sum())
    # print(scales.amax(dim=0),opacity.amax(dim=0))
    # print(scales.amin(dim=0),opacity.amin(dim=0))
    # print(opacity.mean(),opacity.median())
    # print(shs,rotations)
    # print(means3D,rotations,shs)
    # if pc.is_fine:
    #     means3D=means3D[~pc.dynamatic_mask]
    #     rotations=rotations[~pc.dynamatic_mask]
    #     scales=scales[~pc.dynamatic_mask]
    #     opacity=opacity[~pc.dynamatic_mask]
    #     shs=shs[~pc.dynamatic_mask]
    #     means2D=means2D[~pc.dynamatic_mask]
    # else:
    #     means3D = pc._xyz
    #     shs = pc.get_features

    rendered_image, radii,depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    # print(radii)
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
             "depth": depth
            }

def test_ours_flow_mlp_opacity(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    基本完全参照gs-flow的实现
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means.这个东西只是用来记录梯度的，自己的值为0
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    print(torch.cuda.is_available())
    startime = time.time()
    torch.cuda.synchronize()

    # pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # bg_color = torch.tensor([1,1,1], dtype=torch.float32, device="cuda")
    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False
        # ,debug=pipe.debug
        )

    rasterizer = GRzer(raster_settings=raster_settings)

    print("kk")

    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity


    
    opacity = pointopacity   # - 0.5
    # pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling
    shs = pc.get_features
    print(scales.mean(dim=0))
    colors_precomp = None
    # means3D,rotations,scales,opacity = pc.get_deformation(viewpoint_camera.timestamp)

    means3D,rotations,scales,opacity,shs,_ = pc.get_deformation(viewpoint_camera.timestamp)
    print("mean3d",means3D.max(dim=0),means3D.min(dim=0))
    # print(scales.max(),scales.min())
    # print(pc.dynamatic)
    # print(pc.dynamatic.mean(dim=0),pc.dynamatic.amax())
    # colors_precomp = pc.get_trbfscale.detach().expand(-1,3) 
    # colors_precomp = pc.get_trbfcenter.detach().expand(-1,3) 
    # colors_precomp = pc.get_opacity.detach().expand(-1,3) 
    # shs=None
    print(pc._trbf_scale[pc._trbf_scale<3.3e-3].shape)
    intergral = pc.get_intergral()
    print(intergral[pc._trbf_scale<3.3e-3])
    print(pc._trbf_center[intergral == 0],pc._trbf_scale[intergral == 0],pc._trbf_scale[intergral == 0].shape)
    print(intergral[intergral>0].min(),pc.get_intergral().max())
    # rotations = pc.get_rotation
    # scales = pc.get_scaling
    print("scales",scales.max(),scales.min())
    # opacity = pointopacity
    # means3D = pc.get_xyz
    # print(means3D,rotations,scales,opacity)
    # print(means3D.shape,rotations.shape,scales.shape,opacity.shape,means2D.shape,shs.shape)
    # for i in range(means3D.shape[0]):
    #     a=means3D[i]
    #     a=rotations[i]
    #     a=scales[i]
    #     a=opacity[i]
    #     a=shs[i]
    #     a=means2D[i]
    # opacity = pc.get_opacity_deform(viewpoint_camera.timestamp)
    # print(pc.start_end)
    # print(pc._xyz,means3D)
    # means3D = pc._xyz
    # print(pc._xyz,means3D)
    # select_mask = (torch.logical_and(pc._xyz[:,0] >12,pc._xyz[:,2] >0)).squeeze()
    intergral = pc.get_intergral()

    # print(pc._trbf_scale.min())
    # print(pc._xyz.max(dim=0),pc._xyz.min(dim=0),pc._xyz.shape)
    # select_mask = (pc._xyz[:,2] >200).squeeze()
    # print("sssum",(pc.get_opacity[~pc.dynamatic_mask]<0.005).sum()) 
    # select_mask = torch.logical_and(pc._trbf_scale <0.5,intergral<0.1).squeeze()    # select_mask = (torch.logical_and(pc.xyz[:,2] >45,pc._xyz[:,2] <175)).squeeze()
    # select_mask = (scales.max(dim=1).values > 2).squeeze()
    # select_mask = (intergral <0.5).squeeze()
    # # select_mask = (pc._trbf_scale <0.5).squeeze()
    # # select_mask = ~select_mask
    # print(intergral[select_mask].min(),intergral[select_mask].max())
    # print(select_mask.sum())
    # print("bounds",means3D.max(dim=0),means3D.min(dim=0),means3D.shape)
    # bounds = means3D[select_mask]
    # print(bounds.mean(dim=0),bounds.std(dim=0))
    # print(bounds.max(dim=0).values,bounds.min(dim=0).values)
    # select_mask = torch.logical_and(torch.all(means3D<bounds.max(dim=0).values,dim=1),torch.all(means3D>bounds.min(dim=0).values,dim=1)).squeeze()
    # select_mask = ~pc.dynamatic_mask
    # select_mask = ~select_mask
    # select_mask = (pc._xyz[:,2]>4.5).squeeze()
    # means3D = means3D[select_mask]
    # # means3D = pc._xyz[select_mask]

    # print("now bounds",means3D.max(dim=0),means3D.min(dim=0))
    # # print(pc._trbf_center[select_mask].max(dim=0),pc._trbf_center[select_mask].min(dim=0),means3D.shape)
    # means2D = means2D[select_mask]
    # shs = shs[select_mask]
    # print(opacity.shape)
    # opacity = opacity[select_mask]
    # print(opacity.shape)
    # scales = scales[select_mask]
    # rotations = rotations[select_mask]
        # means3D = pc._xyz
    rendered_image, radii,depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    print(torch.cuda.is_available())

    torch.cuda.synchronize()
    duration = time.time() - startime

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth,
            "duration": duration
            }
def train_ours_flow_mlp_opacity_color(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, GRsetting=None, GRzer=None, stage=None):
    """
    Render the scene. 
    基本完全参照gs-flow的实现
    """
    if stage == None:
        raise Exception('没传stage')
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means.这个东西只是用来记录梯度的，自己的值为0
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False
        # ,debug=pipe.debug
        )

    rasterizer = GRzer(raster_settings=raster_settings)

    
    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity


    
    opacity = pointopacity   # - 0.5
    # pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling

    shs = None
    colors_precomp = None
    rotations = pc.get_rotation
    # print(scales,rotations)
    # print(pc.timescale.is_leaf,"dd")
    # print(stage)
    if stage == "dynamatic":
        if pc.args.onemlp:
            means3D,rotations,scales,opacity,shs,colors_precomp = pc.get_cas_deformation(viewpoint_camera.timestamp)
        else:
            means3D,rotations,scales,opacity,_,colors_precomp = pc.get_deformation(viewpoint_camera.timestamp)
        
        # opacity = pc.get_opacity_deform(viewpoint_camera.timestamp)
        # means3D.requires_grad = True
    # print(means3D.shape,rotations.shape,scales.shape,opacity.shape,means2D.shape,shs.shape)
    # # print(means3D,opacity,scales)
    # print(means3D.isnan().sum(),rotations.isnan().sum(),scales.isnan().sum(),opacity.isnan().sum(),shs.isnan().sum(),means2D.isnan().sum())
    # print(scales.amax(dim=0),opacity.amax(dim=0))
    # print(scales.amin(dim=0),opacity.amin(dim=0))
    # print(opacity.mean(),opacity.median())
    # print(shs,rotations)
    # print(means3D,rotations,shs)
    rendered_image, radii,depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    # print(radii)
    rendered_image = pc.get_rgb(rendered_image.unsqueeze(0), viewpoint_camera.rays, viewpoint_camera.timestamp) # 1 , 3 #在这里才把9维的feature逐像素的变成RGB color
    rendered_image = rendered_image.squeeze(0)
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
             "depth": depth
            }
def train_ours_bi_flow(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None, stage=None):
    """
    Render the scene. 
    基本完全参照gs-flow的实现
    """
    if stage == None:
        raise Exception('没传stage')
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means.这个东西只是用来记录梯度的，自己的值为0
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False
        # ,debug=pipe.debug
        )

    rasterizer = GRzer(raster_settings=raster_settings)

    
    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity


    
    opacity = pointopacity   # - 0.5
    # pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling

    colors_precomp = None
    rotations = pc.get_rotation
    # print(scales,rotations)
    # print(pc.timescale.is_leaf,"dd")
    # print(stage)
    valid_mask =None
    key_index_tuple =None
    if stage == "dynamatic":

        means3D,rotations,scales,opacity,shs,valid_mask,key_index_tuple = pc.get_deformation(viewpoint_camera.timestamp)
        # opacity
        # print(key_index_tuple)
        screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
            # print(123)
        except:
            pass
        # print("hhh")
        means2D = screenspace_points
        # means3D=torch.zeros_like(means3D, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
        # rotations=torch.zeros_like(rotations,dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
        # scales=torch.zeros_like(means3D, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
        # opacity=torch.zeros_like(opacity, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0.015
        # shs = torch.zeros_like(shs, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
    else:
        shs = pc.get_features

    # print(means3D.shape,means2D.shape,shs.shape,opacity.shape,rotations.shape,scales.shape)
    print(opacity)
        # opacity = pc.get_opacity_deform(viewpoint_camera.timestamp)
        # means3D.requires_grad = True
    # print(shs,rotations)
    # print(means3D,rotations,shs)
    rendered_image, radii,depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    # print(screenspace_points.grad)
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
             "depth": depth,
             "valid_mask":valid_mask,
             "key_index":key_index_tuple
            }

def test_ours_bi_flow(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    基本完全参照gs-flow的实现
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means.这个东西只是用来记录梯度的，自己的值为0
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    print(torch.cuda.is_available())
    startime = time.time()
    torch.cuda.synchronize()

    # pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False
        # ,debug=pipe.debug
        )

    rasterizer = GRzer(raster_settings=raster_settings)

    print("kk")

    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity


    
    opacity = pointopacity   # - 0.5
    # pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling
    # prune_mask = (opacity < 5e-2).squeeze()
    # print(prune_mask)
    # shs = pc.get_features
    # print(scales)
    colors_precomp = None
    # means3D,rotations,scales,opacity = pc.get_deformation(viewpoint_camera.timestamp)
    means3D,rotations,scales,opacity,shs,_,_ = pc.get_deformation(viewpoint_camera.timestamp)
    # print(pc.dynamatic)
    # print(pc.dynamatic.mean(dim=0),pc.dynamatic.amax())
    # colors_precomp = pc.get_opacity.detach().expand(-1,3) 
    # shs=None
    # rotations = pc.get_rotation
    # scales = pc.get_scaling
    # opacity = pointopacity
    # means3D = pc.get_xyz
    # print(means3D,rotations,scales,opacity)
    # print(means3D.shape,rotations.shape,scales.shape,opacity.shape,means2D.shape,shs.shape)
    # for i in range(means3D.shape[0]):
    #     a=means3D[i]
    #     a=rotations[i]
    #     a=scales[i]
    #     a=opacity[i]
    #     a=shs[i]
    #     a=means2D[i]
    # opacity = pc.get_opacity_deform(viewpoint_camera.timestamp)
    # print(pc.start_end)
    # print(pc._xyz,means3D)
    # means3D = pc._xyz
    # print(pc._xyz,means3D)
    
    # means3D = pc._xyz
    rendered_image, radii,depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    print(torch.cuda.is_available())

    torch.cuda.synchronize()
    duration = time.time() - startime

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth,
            "duration": duration
            }

def train_ours_bi_flow_alldeform(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None, stage=None):
    """
    Render the scene. 
    基本完全参照gs-flow的实现
    """
    if stage == None:
        raise Exception('没传stage')
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means.这个东西只是用来记录梯度的，自己的值为0
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False
        # ,debug=pipe.debug
        )

    rasterizer = GRzer(raster_settings=raster_settings)

    
    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity


    
    opacity = pointopacity   # - 0.5
    # pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling

    colors_precomp = None
    rotations = pc.get_rotation
    # print(scales,rotations)
    # print(pc.timescale.is_leaf,"dd")
    # print(stage)
    valid_mask =None
    key_index_tuple =None
    if stage == "dynamatic":

        means3D,rotations,scales,opacity,shs = pc.get_all_deform(viewpoint_camera.timestamp)
        # opacity
        # print(key_index_tuple)
        screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
            # print(123)
        except:
            pass
        # print("hhh")
        means2D = screenspace_points
        # means3D=torch.zeros_like(means3D, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
        # rotations=torch.zeros_like(rotations,dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
        # scales=torch.zeros_like(means3D, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
        # opacity=torch.zeros_like(opacity, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0.015
        # shs = torch.zeros_like(shs, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
    else:
        shs = pc.get_features

    # print(means3D.shape,means2D.shape,shs.shape,opacity.shape,rotations.shape,scales.shape)
    # print(opacity)
        # opacity = pc.get_opacity_deform(viewpoint_camera.timestamp)
        # means3D.requires_grad = True
    # print(shs,rotations)
    # print(means3D,rotations,shs)
    rendered_image, radii,depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    # print(screenspace_points.grad)
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
             "depth": depth
            }

def test_ours_bi_flow_alldeform(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    基本完全参照gs-flow的实现
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means.这个东西只是用来记录梯度的，自己的值为0
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    print(torch.cuda.is_available())
    startime = time.time()
    torch.cuda.synchronize()

    # pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False
        # ,debug=pipe.debug
        )

    rasterizer = GRzer(raster_settings=raster_settings)

    print("kk")

    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity


    
    opacity = pointopacity   # - 0.5
    # pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling
    # prune_mask = (opacity < 5e-2).squeeze()
    # print(prune_mask)
    # shs = pc.get_features
    # print(scales)
    colors_precomp = None
    # means3D,rotations,scales,opacity = pc.get_deformation(viewpoint_camera.timestamp)
    means3D,rotations,scales,opacity,shs = pc.get_all_deform(viewpoint_camera.timestamp)
    # print(pc.dynamatic)
    # print(pc.dynamatic.mean(dim=0),pc.dynamatic.amax())
    # colors_precomp = pc.get_opacity.detach().expand(-1,3) 
    # shs=None
    # rotations = pc.get_rotation
    # scales = pc.get_scaling
    # opacity = pointopacity
    # means3D = pc.get_xyz
    # print(means3D,rotations,scales,opacity)
    # print(means3D.shape,rotations.shape,scales.shape,opacity.shape,means2D.shape,shs.shape)
    # for i in range(means3D.shape[0]):
    #     a=means3D[i]
    #     a=rotations[i]
    #     a=scales[i]
    #     a=opacity[i]
    #     a=shs[i]
    #     a=means2D[i]
    # opacity = pc.get_opacity_deform(viewpoint_camera.timestamp)
    # print(pc.start_end)
    # print(pc._xyz,means3D)
    # means3D = pc._xyz
    # print(pc._xyz,means3D)
    
    # means3D = pc._xyz
    rendered_image, radii,depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    print(torch.cuda.is_available())

    torch.cuda.synchronize()
    duration = time.time() - startime

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth,
            "duration": duration
            }
