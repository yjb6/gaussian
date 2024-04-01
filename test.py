# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ========================================================================================================
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the thirdparty/gaussian_splatting/LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import sys 
sys.path.append("./thirdparty/gaussian_splatting")

import torch
from thirdparty.gaussian_splatting.scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import torchvision
import time 
import scipy
import numpy as np 
import warnings
import json 
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image

from thirdparty.gaussian_splatting.lpipsPyTorch import lpips
from helper_train import getrenderpip, getmodel, trbfunction
from thirdparty.gaussian_splatting.utils.loss_utils import ssim
from thirdparty.gaussian_splatting.utils.image_utils import psnr
from thirdparty.gaussian_splatting.helper3dg import gettestparse
from skimage.metrics import structural_similarity as sk_ssim
from thirdparty.gaussian_splatting.arguments import ModelParams, PipelineParams
from helper_test import rgbd2pcd, calculate_trajectories, get_k,calculate_rot_vec

warnings.filterwarnings("ignore")

# modified from https://github.com/graphdeco-inria/gaussian-splatting/blob/main/render.py and https://github.com/graphdeco-inria/gaussian-splatting/blob/main/metrics.py
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, rbfbasefunction, rdpip,frame_duration,args):
    print(rdpip)
    # render, GRsetting, GRzer = getrenderpip(rdpip) 
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_xyz")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    traj_path = os.path.join(model_path, name, "ours_{}".format(iteration), "traj")
    rot_path = os.path.join(model_path, name, "ours_{}".format(iteration), "rot")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(traj_path, exist_ok=True)
    makedirs(rot_path, exist_ok=True)
    try:
        if gaussians.rgbdecoder is not None:
            gaussians.rgbdecoder.cuda()
            gaussians.rgbdecoder.eval()
    except:
        print("no rgb decoder")
    statsdict = {}

    scales = gaussians.get_scaling

    scalemax = torch.amax(scales).item()
    scalesmean = torch.amin(scales).item()
     
    op = gaussians.get_opacity
    opmax = torch.amax(op).item()
    opmean = torch.mean(op).item()

    statsdict["scales_max"] = scalemax
    statsdict["scales_mean"] = scalesmean

    statsdict["op_max"] = opmax
    statsdict["op_mean"] = opmean 


    statspath = os.path.join(model_path, "stat_" + str(iteration) + ".json")
    with open(statspath, 'w') as fp:
            json.dump(statsdict, fp, indent=True)


    psnrs = []
    lpipss = []
    lpipssvggs = []

    full_dict = {}
    per_view_dict = {}
    ssims = []
    ssimsv2 = []
    scene_dir = model_path
    image_names = []
    times = []

    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}

  

    full_dict[scene_dir][iteration] = {}
    per_view_dict[scene_dir][iteration] = {}


    if rdpip == "train_ours_full":
        render, GRsetting, GRzer = getrenderpip("test_ours_full") 
    elif rdpip == "train_ours_lite":
        render, GRsetting, GRzer = getrenderpip("test_ours_lite") 
    elif rdpip == "train_ours_flow":
        render, GRsetting, GRzer = getrenderpip("test_ours_flow")
    elif rdpip == "train_ours_flow_full":
        render, GRsetting, GRzer = getrenderpip("test_ours_flow_full")  
    elif rdpip == "train_ours_flow_mlp":
        render, GRsetting, GRzer = getrenderpip("test_ours_flow_mlp")
    elif rdpip == "train_ours_xyz_mlp":
        render, GRsetting, GRzer = getrenderpip("test_ours_xyz_mlp")
    elif rdpip == "train_ours_flow_mlp_opacity":
        render, GRsetting, GRzer = getrenderpip("test_ours_flow_mlp_opacity")
    elif rdpip == "train_ours_bi_flow":
        render, GRsetting, GRzer = getrenderpip("test_ours_bi_flow")
    elif rdpip == "train_ours_bi_flow_alldeform":
        render, GRsetting, GRzer = getrenderpip("test_ours_bi_flow_alldeform")
    else:
        render, GRsetting, GRzer = getrenderpip(rdpip) 


    for idx, view in enumerate(tqdm(views, desc="Rendering and metric progress")):
        # print(view.image_width,view.image_height)
        # view = views[-1]
        renderingpkg = render(view, gaussians, pipeline, background, scaling_modifier=1.0, basicfunction=rbfbasefunction,  GRsetting=GRsetting, GRzer=GRzer) # C x H x W
        rendering = renderingpkg["render"]
        rendering = torch.clamp(rendering, 0, 1.0)
        if "depth" in renderingpkg:
            depth = renderingpkg["depth"]
            depth_np = renderingpkg["depth"].squeeze().detach().cpu().numpy()
            plt.imsave(os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"), depth_np, cmap='viridis')

        if args.traj  and idx>0 and idx % args.traj_interval==0:
            motion_save_path = os.path.join(traj_path, '{0:05d}'.format(idx) + ".png")
            rot_save_path = os.path.join(rot_path, '{0:05d}'.format(idx) + ".png")

            traj_length = min(idx,args.max_traj_length) 
            visualize_traj(gaussians,view,rendering,depth,idx,frame_duration,motion_save_path,type="motion",traj_length=traj_length,traj_frac=args.traj_frac)
            visualize_traj(gaussians,view,rendering,depth,idx,frame_duration,rot_save_path,type="rot",traj_length=traj_length,traj_frac=args.traj_frac)

        # print(depth_np.shape)
        # print(depth.max(), depth.min())
        # depth = depth/(depth.max() - depth.min())
        

        gt = view.original_image[0:3, :, :].cuda().float()
        ssims.append(ssim(rendering.unsqueeze(0),gt.unsqueeze(0))) 

        psnrs.append(psnr(rendering.unsqueeze(0), gt.unsqueeze(0)))
        lpipss.append(lpips(rendering.unsqueeze(0), gt.unsqueeze(0), net_type='alex')) #
        lpipssvggs.append( lpips(rendering.unsqueeze(0), gt.unsqueeze(0), net_type='vgg'))

        rendernumpy = rendering.permute(1,2,0).detach().cpu().numpy()
        gtnumpy = gt.permute(1,2,0).detach().cpu().numpy()
        # print(rendernumpy.shape)
        # print(gtnumpy.shape)
        ssimv2 =  sk_ssim(rendernumpy, gtnumpy, multichannel=True, data_range=1.0,channel_axis=-1)
        ssimsv2.append(ssimv2)


        # print(depth,depth.shape)
        # print(gt)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        image_names.append('{0:05d}'.format(idx) + ".png")

    

    for idx, view in enumerate(tqdm(views, desc="release gt images cuda memory for timing")):
        view.original_image = None #.detach()  
        torch.cuda.empty_cache()

    # start timing
    for _ in range(4):
        for idx, view in enumerate(tqdm(views, desc="timing ")):

            renderpack = render(view, gaussians, pipeline, background, scaling_modifier=1.0, basicfunction=rbfbasefunction,  GRsetting=GRsetting, GRzer=GRzer)#["time"] # C x H x W
            duration = renderpack["duration"]
            if idx > 10: #warm up
                times.append(duration)

    print(np.mean(np.array(times)))
    if len(views) > 0:
        full_dict[model_path][iteration].update({"SSIM": torch.tensor(ssims).mean().item(),
                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                        "LPIPS": torch.tensor(lpipss).mean().item(),
                                        "ssimsv2": torch.tensor(ssimsv2).mean().item(),
                                        "LPIPSVGG": torch.tensor(lpipssvggs).mean().item(),
                                        "times": torch.tensor(times).mean().item()})
        
        per_view_dict[model_path][iteration].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                                "ssimsv2": {name: v for v, name in zip(torch.tensor(ssimsv2).tolist(), image_names)},
                                                                "LPIPSVGG": {name: lpipssvgg for lpipssvgg, name in zip(torch.tensor(lpipssvggs).tolist(), image_names)},})
        
            
            
        with open(model_path + "/" + str(iteration) + "_runtimeresults.json", 'w') as fp:
            json.dump(full_dict, fp, indent=True)

        with open(model_path + "/" + str(iteration) + "_runtimeperview.json", 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)

def visualize_traj(gaussians,cam, image, depth,frame_idx,duration, save_path,type="motion",traj_length=20,traj_frac =25):
    '''traj_frac表示记录百分之多少的点。traj_length表示记录的轨迹长度'''
    w, h = cam.image_width,cam.image_height
    render_numpy = (image.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    view_scale = 1.0
    RENDER_MODE = 'color'
    renderer = o3d.visualization.rendering.OffscreenRenderer(w,h)

    K = get_k(cam)
    w2c = cam.world_view_transform.T
    # # 创建 PinholeCameraIntrinsic 对象
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(int(w), int(h), K[0, 0], K[1, 1], int(K[0, 2]), int(K[1, 2]))
    #创建material
    material = o3d.cuda.pybind.visualization.rendering.MaterialRecord()

    timelist = np.arange(0, frame_idx+1)/duration
    if type == "motion":
        linesets = calculate_trajectories(gaussians, timelist,traj_length,traj_frac)
    elif type == "rot":
        linesets = calculate_rot_vec(gaussians, timelist,traj_length,traj_frac)
    else:
        raise ValueError("type must be motion or rotation")
    # elif ADDITIONAL_LINES == 'rotations': 旋转的还没写
    #     linesets = calculate_rot_vec(scene_data, is_fg)
    lines = o3d.geometry.LineSet()
    lines.points = linesets[-1].points
    lines.colors = linesets[-1].colors
    lines.lines = linesets[-1].lines
    lines.colors = linesets[0].colors
    lines.lines = linesets[0].lines
    renderer.scene.scene.add_geometry("lines",lines,material)

    renderer.setup_camera(intrinsics, w2c.cpu().numpy())
    backgoround=o3d.geometry.Image(np.ascontiguousarray(np.asarray(image.permute(1,2,0).cpu()))) #将原图像作为背景图像加入
    renderer.scene.set_background(np.array([0,0,0,0]),backgoround)
    image = renderer.render_to_image()

    return o3d.io.write_image(save_path, image, quality=9)


# render free view
def render_setnogt(model_path, name, iteration, views, gaussians, pipeline, background, rbfbasefunction, rdpip):
    render, GRsetting, GRzer = getrenderpip(rdpip) 
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")

    makedirs(render_path, exist_ok=True)
    if gaussians.rgbdecoder is not None:
        gaussians.rgbdecoder.cuda()
        gaussians.rgbdecoder.eval()

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        rendering = render(view, gaussians, pipeline, background,scaling_modifier=1.0, basicfunction=rbfbasefunction,  GRsetting=GRsetting, GRzer=GRzer)["render"] # C x H x W

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))


def run_test(dataset : ModelParams, ckpt, pipeline : PipelineParams, skip_train : bool, skip_test : bool, multiview : bool, duration: int,args ,rgbfunction="rgbv1", rdpip="v2", loader="colmap"):
    
    with torch.no_grad():
        print("use model {}".format(dataset.model))
        GaussianModel = getmodel(dataset.model) # default, gmodel, we are tewsting 

        gaussians = GaussianModel(dataset, rgbfunction=rgbfunction)
        gaussians.duration = args.duration
        gaussians.preprocesspoints = args.preprocesspoints 

        if dataset.color_order >0:
            gaussians.color_order = dataset.color_order
            
        # scene = Scene(dataset, gaussians, load_iteration=ckpt,shuffle=False, multiview=multiview, duration=duration, loader=loader)
        # iteration = ckpt
        scene = Scene(dataset, gaussians, shuffle=False, multiview=multiview, duration=duration, loader=loader)
        rbfbasefunction = trbfunction
        numchannels = 9
        bg_color =  [0 for _ in range(numchannels)]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        (model_params, iteration) = torch.load(ckpt)
        print("load at",iteration)
        gaussians.restore(model_params, None)
        if hasattr(gaussians,"ts") and  gaussians.ts is None :
            cameraslit = scene.getTestCameras()
            H,W = cameraslit[0].image_height, cameraslit[0].image_width
            gaussians.ts = torch.ones(1,1,H,W).cuda()

        if not skip_test and not multiview:            
            render_set(dataset.model_path, "test", iteration, scene.getTestCameras(), gaussians, pipeline, background, rbfbasefunction, rdpip,frame_duration=duration,args=args)
        if multiview:
            render_setnogt(dataset.model_path, "mv", iteration, scene.getTestCameras(), gaussians, pipeline, background, rbfbasefunction, rdpip)

if __name__ == "__main__":
    

    args, model_extract, pp_extract, multiview =gettestparse()
    # run_test(model_extract, args.test_iteration, pp_extract, args.skip_train, args.skip_test, multiview, args.duration,  args,rgbfunction=args.rgbfunction, rdpip=args.rdpip, loader=args.valloader)
    run_test(model_extract, args.checkpoint, pp_extract, args.skip_train, args.skip_test, multiview, args.duration,  args,rgbfunction=args.rgbfunction, rdpip=args.rdpip, loader=args.valloader)