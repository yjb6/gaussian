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

import os
import torch
import traceback
from random import randint
import random 
import sys 
import uuid
import time 
import json
import wandb
import torchvision
import numpy as np 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import cv2
from tqdm import tqdm
import time
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
sys.path.append("./thirdparty/gaussian_splatting")
from thirdparty.gaussian_splatting.utils.system_utils import mkdir_p
from thirdparty.gaussian_splatting.utils.loss_utils import l1_loss, ssim, l2_loss, rel_loss,msssim
from thirdparty.gaussian_splatting.utils.image_utils import psnr,easy_cmap
from helper_train import getrenderpip, getmodel, getloss, controlgaussians, reloadhelper, trbfunction
from thirdparty.gaussian_splatting.scene import Scene
from thirdparty.gaussian_splatting.scene.dataset import IdxDataset,SameTimeDataLoader,KeyIndexDataLoader,TimeBatchDataset
from argparse import Namespace
from thirdparty.gaussian_splatting.helper3dg import getparser, getrenderparts


def train(dataset, opt, pipe, saving_iterations,testing_iterations, debug_from,checkpoint = None ,densify=0, duration=50,wandb_run = None, rgbfunction="rgbv1", rdpip="v2"):
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    first_iter = 0
    # render, GRsetting, GRzer = getrenderpip(rdpip)
    render = getrenderpip(rdpip)

    print("use model {}".format(dataset.model))
    GaussianModel = getmodel(dataset.model) # gmodel, gmodelrgbonly
    
    gaussians = GaussianModel(dataset, rgbfunction=rgbfunction)
    gaussians.trbfslinit = -1*opt.trbfslinit # spt中给scale做初始化的
    gaussians.preprocesspoints = opt.preprocesspoints 
    gaussians.addsphpointsscale = opt.addsphpointsscale 
    gaussians.raystart = opt.raystart
    gaussians.duration = duration
    if dataset.color_order >0:
        gaussians.color_order = dataset.color_order

    rbfbasefunction = trbfunction

    scene = Scene(dataset, gaussians, duration=duration, loader=dataset.loader)
    gaussians.training_setup(opt)
    print(checkpoint)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        if first_iter >= opt.static_iteration:#已经进行过转换了
            gaussians.is_dynamatic = True

    currentxyz = gaussians._xyz 
    maxx, maxy, maxz = torch.amax(currentxyz[:,0]), torch.amax(currentxyz[:,1]), torch.amax(currentxyz[:,2])# z wrong...
    minx, miny, minz = torch.amin(currentxyz[:,0]), torch.amin(currentxyz[:,1]), torch.amin(currentxyz[:,2])
     

    if os.path.exists(opt.prevpath):
        print("load from " + opt.prevpath)
        reloadhelper(gaussians, opt, maxx, maxy, maxz,  minx, miny, minz)
   


    maxbounds = [maxx, maxy, maxz]
    minbounds = [minx, miny, minz]


    numchannel = 9 

    bg_color = [1, 1, 1] if dataset.white_background else [0 for i in range(numchannel)]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_l1loss_for_log = 0.0
    best_psnr = 0.0
    history_data=None

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    # first_iter += 1

    flag = 0
    flagtwo = 0
    depthdict = {}

    if opt.batch > 1 and opt.multiview:
        #针对多目数据集，记录同一时刻的相机列表
        
        traincameralist = scene.getTrainCamInfos().copy() if dataset.use_loader else scene.getTrainCameras().copy()
        # traincamerainfolist = scene.getTrainCamInfos().copy()
        traincamdict = {}
        for i in range(duration): # 0 to 4, -> (0.0, to 0.8)
            if dataset.use_loader:
                if opt.sametime_batch or opt.keyindex_batch:
                    #每次采样出相同时间的数据
                    traincamdict[i] = [idx for idx,cam in enumerate(traincameralist) if cam.timestamp == i/duration] #如果是用loader，就记录下标

                else:
                    #random
                    traincam_dataset = scene.getTrainCameras()

            else:
                
                traincamdict[i] = [cam for cam in traincameralist if cam.timestamp == i/duration]
        if dataset.use_loader:
            if opt.keyindex_batch:
                # idx_dataset = IdxDataset(traincamdict)
                time_dataset = TimeBatchDataset(traincamdict,gaussians.key_frame_dict,1,scene)
                loader = DataLoader(time_dataset,batch_size=2,num_workers=2,shuffle=True,collate_fn=list)
                # for data in loader:
                #     print(data)
                #     time.sleep(2)
                #     print("wake up")
            elif opt.sametime_batch:
                print("samttime batchs")
                #每次采样出相同时间的数据
                idx_dataset = IdxDataset(traincamdict)
                loader = SameTimeDataLoader(idx_dataset,opt.batch,scene)
                test_loader = DataLoader(scene.getTestCameras(), batch_size=1,shuffle=False,num_workers=32,collate_fn=lambda x: x)
                # while True:
                #     for data in loader:
                #         print(data)

            else:
                loader = DataLoader(traincam_dataset, batch_size=opt.batch,shuffle=True,num_workers=32,collate_fn=list)
                test_loader = DataLoader(scene.getTestCameras(), batch_size=1,shuffle=False,num_workers=32,collate_fn=lambda x: x)
                # while True:
                #     for data in loader:
                #         print(data)
            # loader = iter(loader)

    elif opt.batch ==1 and not opt.multiview:
        traincameralist = scene.getTrainCameras().copy()
        # traincamdict = {}
        # for cam in traincameralist:
        #     traincamdict
    if hasattr(gaussians,"ts") and gaussians.ts is None :
        H,W = scene.getTrainCameras()[0].image_height, scene.getTrainCameras()[0].image_width 
        gaussians.ts = torch.ones(1,1,H,W).cuda()

    scene.recordpoints(0, "start training")

                                                            
    flagems = 0  
    emscnt = 0
    lossdiect = {}
    ssimdict = {}
    depthdict = {}
    validdepthdict = {}
    emsstartfromiterations = opt.emsstart   
    # if opt.multiview:
    #     with torch.no_grad():
    #         timeindex = 0 # 0 to 49
    #         viewpointset = traincamdict[timeindex]
    #         for viewpoint_cam in viewpointset:
    #             # print(viewpoint_cam)
    #             if dataset.use_loader:
    #                 # print(viewpoint_cam)
    #                 viewpoint_cam = scene.getTrainCameras()[viewpoint_cam]
    #             # print(gaussians.timescale.is_leaf,"dd")
    #             render_pkg = render(viewpoint_cam, gaussians, pipe, background,  override_color=None,  basicfunction=rbfbasefunction, GRsetting=GRsetting, GRzer=GRzer,stage="static")
                
    #             # _, depthH, depthW = render_pkg["depth"].shape
    #             # borderH = int(depthH/2)
    #             # borderW = int(depthW/2)

    #             midh =  int(viewpoint_cam.image_height/2)
    #             midw =  int(viewpoint_cam.image_width/2)
                
                # depth = render_pkg["depth"]
                # slectemask = depth != 15.0 

                # validdepthdict[viewpoint_cam.image_name] = torch.median(depth[slectemask]).item()   
                # depthdict[viewpoint_cam.image_name] = torch.amax(depth[slectemask]).item() 
    
    if (densify == 1 or  densify == 2 )and not dataset.random_init: 
        #这个过滤对减少悬浮物非常的重要
        zmask = gaussians._xyz[:,2] < 4.5  

        gaussians.prune_points(zmask) 
        print("After pure z<4.5",gaussians._xyz.shape[0])
        torch.cuda.empty_cache()


    selectedlength = 2
    lasterems = 0 

    lambda_all = [key for key in opt.__dict__.keys() if key.startswith('lambda') ] #记录所有的loss
    for lambda_name in lambda_all:
        vars()[f"ema_{lambda_name.replace('lambda_','')}_for_log"] = 0.0
    


    # for iteration in range(first_iter,opt.iterations+1):
    iteration = first_iter
    # gaussians.reset_opacity()
    # print("reset opacity")
    print(gaussians.get_scaling.max(),gaussians.get_scaling.mean(),gaussians.get_scaling.min())
    # exit()
    while iteration < opt.iterations+1:
        # print(iteration)
        for camindex in loader: #统一使用dataloder
            iteration +=1
            if iteration > opt.iterations:
                break
            if opt.coarse_iteration >=0 and iteration > opt.coarse_iteration:
                if not gaussians.is_fine:
                    gaussians.coarse2fine()
                    opt.lambda_dscale_entropy = 0
            if iteration > opt.static_iteration:
                stage = "dynamatic"
                if  not gaussians.is_dynamatic:
                    gaussians.static2dynamatic()
            else:
                stage = "static" 
            if iteration ==  opt.emsstart:
                flagems = 1 # start ems . 并且这个ems是只进行一次的

            iter_start.record()
            gaussians.update_learning_rate(iteration,stage=stage)
            
            if (iteration - 1) == debug_from:
                pipe.debug = True
            # if gaussians.rgbdecoder is not None:
            #     gaussians.rgbdecoder.train()

            if opt.batch > 1 and opt.multiview:
                gaussians.zero_gradient_cache()
                # if stage != "dynamatic":
                #     timeindex = 0
                # else:
                # timeindex = randint(0, duration-1) # 0 to 49
                # viewpointset = traincamdict[timeindex]
                # # print(viewpointset,opt.batch)
                # camindex = random.sample(viewpointset, opt.batch)

                # for i in range(opt.batch):
                #     viewpoint_cam = camindex[i]
                # if not dataset.use_loader: #如果没有用dataloader，则改变
                #     timeindex = randint(0, duration-1) # 0 to 49
                #     viewpointset = traincamdict[timeindex]
                #     camindex = random.sample(viewpointset, opt.batch)
                # else:
                #     try:
                #         camindex = next(loader)
                #         if opt.keyindex_batch:
                #             camindex = camindex[0]
                #     except StopIteration:
                #         print("reset dataloader into random dataloader.")
                #         if opt.keyindex_batch:
                #             # idx_dataset = IdxDataset(traincamdict)
                #             loader = DataLoader(time_dataset,batch_size=1,num_workers=1,collate_fn=list)

                #             # loader = KeyIndexDataLoader(idx_dataset,gaussians.key_frame_dict,traincameralist,opt.batch,scene)
                #         elif opt.sametime_batch:
                #             #每次采样出相同时间的数据
                #             loader = SameTimeDataLoader(idx_dataset,opt.batch,scene)

                #         else:
                #             #random
                #             loader = DataLoader(traincam_dataset, batch_size=opt.batch,shuffle=True,num_workers=32,collate_fn=list)
                #         loader = iter(loader)
                #         camindex = next(loader)
                #         if opt.keyindex_batch:
                #             camindex = camindex[0]
                batch_point_grad = []
                batch_visibility_filter = []
                batch_radii = []
                for viewpoint_cam in camindex:

                    render_pkg = render(viewpoint_cam, gaussians, pipe, background,stage=stage)
                    image, viewspace_point_tensor, visibility_filter, radii = getrenderparts(render_pkg) 

                    valid_mask = None #当是bi-flow动态阶段时，valid_mask不是None
                    if "valid_mask" in render_pkg:
                        valid_mask = render_pkg["valid_mask"]
                        key_index_tuple = render_pkg["key_index"]
                    gt_image = viewpoint_cam.original_image.float().cuda()

                    Ll1 = l1_loss(image, gt_image)
                    loss,loss_dict = getloss(opt, Ll1, ssim, image, gt_image, gaussians, radii, viewpoint_cam.timestamp,iteration,lambda_all)
                    # if opt.reg == 2:
                    #     Ll1 = l2_loss(image, gt_image)
                    #     loss = Ll1
                    # elif opt.reg == 3:
                    #     Ll1 = rel_loss(image, gt_image)
                    #     loss = Ll1
                    # else:
                    #     Ll1 = l1_loss(image, gt_image)
                    #     loss = getloss(opt, Ll1, ssim, image, gt_image, gaussians, radii, viewpoint_cam.timestamp,iteration)

                    if flagems == 1:
                        if viewpoint_cam.image_name not in lossdiect:
                            #如果lossdict没有保存过这个相机，就加入
                            lossdiect[viewpoint_cam.image_name] = loss.item()
                            ssimdict[viewpoint_cam.image_name] = ssim(image.clone().detach(), gt_image.clone().detach()).item()
                    # print(loss.item())
                    loss.backward()

                    batch_point_grad.append(torch.norm(viewspace_point_tensor.grad[:,:2], dim=-1))
                    batch_radii.append(radii)
                    batch_visibility_filter.append(visibility_filter)
                    # print(viewspace_point_tensor.grad)

                    # if iteration>600:
                    #     print(gaussians._xyz.grad)
                    if stage == "static":
                        try:
                            motion_mlp_out = render_pkg["motion_out"]
                            true_xyz = gaussians.get_xyz.detach()

                            loss_mlp = l2_loss(motion_mlp_out, true_xyz)
                            loss_mlp.backward()
                        except:
                        #     print("no motion_mlp_out")
                            motion_mlp_out = None


                    gaussians.cache_gradient(stage)#把梯度保存下来
                    gaussians.optimizer.zero_grad(set_to_none = True)# 清空梯度
                # print(len(lossdiect.keys()),len(viewpointset))
                # print(lossdiect,ssimdict)

                if flagems == 1 and len(lossdiect.keys()) == len(viewpointset):
                    # sort dict by value
                    orderedlossdiect = sorted(ssimdict.items(), key=lambda item: item[1], reverse=False) # ssimdict lossdiect
                    flagems = 2
                    selectviewslist = []
                    selectviews = {}
                    for idx, pair in enumerate(orderedlossdiect):
                        viewname, lossscore = pair
                        ssimscore = ssimdict[viewname]
                        if ssimscore < 0.91: # avoid large ssim
                            selectviewslist.append((viewname, "rk"+ str(idx) + "_ssim" + str(ssimscore)[0:4]))
                    if len(selectviewslist) < 2 :
                        selectviews = []
                    else:
                        selectviewslist = selectviewslist[:2]
                        for v in selectviewslist:
                            selectviews[v[0]] = v[1]

                    selectedlength = len(selectviews)
                    print(selectviewslist)
                iter_end.record()
                gaussians.set_batch_gradient(opt.batch,stage)
                # exit()
                # note we retrieve the correct gradient except the mask
            elif opt.batch ==1 and not opt.multiview:
                gaussians.zero_gradient_cache()
                if stage == "static":
                    viewpoint_cam = traincameralist[0]
                else:
                    if not traincameralist:
                        traincameralist = scene.getTrainCameras().copy()
                    viewpoint_cam = traincameralist.pop(randint(0, len(traincameralist)-1))
                render_pkg = render(viewpoint_cam, gaussians, pipe, background,  override_color=None,  basicfunction=rbfbasefunction, GRsetting=GRsetting, GRzer=GRzer,stage=stage)
                image, viewspace_point_tensor, visibility_filter, radii = getrenderparts(render_pkg) 
                # print(viewspace_point_tensor,visibility_filter)
                gt_image = viewpoint_cam.original_image.float().cuda()
                if opt.reg == 2:
                    Ll1 = l2_loss(image, gt_image)
                    loss = Ll1
                elif opt.reg == 3:
                    Ll1 = rel_loss(image, gt_image)
                    loss = Ll1
                else:
                    Ll1 = l1_loss(image, gt_image)
                    loss = getloss(opt, Ll1, ssim, image, gt_image, gaussians, radii,viewpoint_cam.timestamp,iteration)
                loss.backward()
                # print(viewspace_point_tensor.grad,visibility_filter)
                # gaussians.cache_gradient()
                # gaussians.optimizer.zero_grad(set_to_none = True)# 
            else:
                raise NotImplementedError("Batch size 1 is not supported")
            if dataset.use_shs :
                if iteration % 1000 == 0:
                    gaussians.oneupSHdegree()
            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_l1loss_for_log = 0.4 * Ll1.item() + 0.6 * ema_l1loss_for_log
                psnr_for_log = psnr(image, gt_image).mean().double()
                for lambda_name in lambda_all:
                    if opt.__dict__[lambda_name] > 0:
                        ema = vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"]
                        vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"] = 0.4 * loss_dict[f"L{lambda_name.replace('lambda_', '')}"].item() + 0.6*ema

                if iteration % 10 == 0:
                    postfix = {"Loss": f"{ema_loss_for_log:.{7}f}",
                                            "PSNR": f"{psnr_for_log:.{2}f}",
                                            "Ll1": f"{ema_l1loss_for_log:.{4}f}",}
                    for lambda_name in lambda_all:
                        if opt.__dict__[lambda_name] > 0:
                            ema_loss = vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"]
                            postfix[lambda_name.replace("lambda_", "L")] = f"{ema_loss:.{4}f}"
                    progress_bar.set_postfix(postfix)

                    # if stage=="static" and motion_mlp_out is not None:
                    #     mlp_loss_for_log = 0.4 * loss_mlp.item() + 0.6 * mlp_loss_for_log
                    #     progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}","MLP_Loss": f"{mlp_loss_for_log:.{7}f}"})
                    # else:
                    #     # print(iteration,loss.item())
                    #     progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})


                    progress_bar.update(10)
                
                if iteration == opt.iterations:
                    progress_bar.close()

                # if iteration%2 ==0:
                # #观测scale和xyz_grad的关系
                #     # print(scene.gaussians.dynamatic_mask.sum())
                #     # print(scene.gaussians._xyz.shape)
                #     # print(scene.gaussians._trbf_scale[~scene.gaussians.dynamatic_mask].min())
                #     scale_torch = scene.gaussians.get_intergral()
                #     # print(scale_torch[~scene.gaussians.dynamatic_mask].min(),scale_torch[~scene.gaussians.dynamatic_mask].max())
                #     print((scale_torch<0.5).sum())
                #     print(scale_torch[(scene.gaussians._trbf_scale>0.9).squeeze()].min())
                #     inv_scale_torch = 1/scale_torch
                #     # inv_scale_torch[inv_scale_torch.isnan()] = 0
                #     print(inv_scale_torch.min(),inv_scale_torch.max())
                #     # print(inv_scale_torch[~scene.gaussians.dynamatic_mask].min(),inv_scale_torch[~scene.gaussians.dynamatic_mask].max())
                #     scale = scene.gaussians.get_intergral().cpu().numpy()
                #     # valid_mask = torch.logical_and(scene.gaussians._trbf_center>=0, scene.gaussians._trbf_center<=1).cpu().numpy()
                #     valid_mask = (scale>=scene.gaussians.min_intergral).squeeze()

                #     scale= scale[valid_mask]
                #     # scale = scene.gaussians._trbf_scale.cpu().numpy()
                #     print(scale.min(),scale.max(),scale.mean(),scale.std(),np.median(scale))
                #     # print(scene.gaussians._trbf_center[scale==0].shape)
                #     space_grad = scene.gaussians.xyz_gradient_accum / scene.gaussians.denom
                #     space_grad[space_grad.isnan()] = 0.0
                #     print((space_grad*inv_scale_torch).min())
                #     # mul_space_grad = scene
                #     space_grad = space_grad.cpu().numpy()[valid_mask]
                #     print(space_grad[np.logical_and(scale>0,space_grad>0)].min())
                #     inv_scale = 1/scale
                #     inv_scale = inv_scale /inv_scale.min()
                #     print(inv_scale.min(),inv_scale.max())
                #     # inv_scale[inv_scale==np.inf] = 0
                #     # inv_scale[inv_scale==np.nan] = 0
                #     space_grad_mul = space_grad *inv_scale
                #     print(space_grad_mul.min(),space_grad_mul.max())
                #     # print(space_grad_mul[np.logical_and(scale>0,space_grad_mul>0)].min())
                #     rot_grad = torch.norm(scene.gaussians._rotation.grad,dim=-1,keepdim=True).cpu().numpy()[valid_mask]
                #     scaling_grad = torch.norm(scene.gaussians._scaling.grad,dim=-1,keepdim=True).cpu().numpy()[valid_mask]
                #     xyz_grad = torch.norm(scene.gaussians._xyz.grad,dim=-1,keepdim=True).cpu().numpy()[valid_mask]
                #     t_center_grad = abs(scene.gaussians._trbf_center.grad.cpu().numpy())[valid_mask]
                #     opacity_grad = abs(scene.gaussians._opacity.grad.cpu().numpy())[valid_mask]
                #     opciaty= scene.gaussians.get_opacity.cpu().numpy()[valid_mask]
                #     t_center = scene.gaussians._trbf_center.cpu().numpy()[valid_mask]


                #     scale_bins = np.linspace(scale.min(),scale.max(),10)
                #     opacity_grad_means = [np.mean(opacity_grad[np.logical_and(scale >= scale_bins[i-1] , scale < scale_bins[i])]) for i in range(len(scale_bins))[1:]]
                #     opacity_means = [np.mean(opciaty[np.logical_and(scale  >= scale_bins[i-1] , scale < scale_bins[i])]) for i in range(len(scale_bins))[1:]]
                #     t_center_means = [np.mean(t_center[np.logical_and(scale  >= scale_bins[i-1] , scale < scale_bins[i])]) for i in range(len(scale_bins))[1:]]
                #     xyz_grad_means = [np.mean(xyz_grad[np.logical_and(scale  >= scale_bins[i-1] , scale < scale_bins[i])]) for i in range(len(scale_bins))[1:]]
                #     # for i in scale_bins:
                #     #     print(i)
                #     space_grad_means = [np.mean(space_grad[np.logical_and(scale  >= scale_bins[i-1] , scale < scale_bins[i])]) for i in range(len(scale_bins))[1:]]
                #     rot_grad_means = [np.mean(rot_grad[np.logical_and(scale  >= scale_bins[i-1] , scale < scale_bins[i])]) for i in range(len(scale_bins))[1:]]
                #     scaling_grad_means = [np.mean(scaling_grad[np.logical_and(scale  >= scale_bins[i-1] , scale < scale_bins[i])]) for i in range(len(scale_bins))[1:]]
                #     t_center_grad_means = [np.mean(t_center_grad[np.logical_and(scale  >= scale_bins[i-1] , scale < scale_bins[i])]) for i in range(len(scale_bins))[1:]]
                #     space_grad_mul_means = [np.mean(space_grad_mul[np.logical_and(scale  >= scale_bins[i-1] , scale < scale_bins[i])]) for i in range(len(scale_bins))[1:]]
                #     print("bins",scale_bins)
                #     print("opacity_grid",opacity_grad_means)
                #     print("opacity",opacity_means)
                #     print("t_center",t_center_means)
                #     print("xyz_grad",xyz_grad_means)
                #     print("space_grad",space_grad_means)
                #     print("space_grad_mul",space_grad_mul_means)
                #     print("rot_grad",rot_grad_means )
                #     print("scaling_grad",scaling_grad_means)
                #     print("t_center_grad",t_center_grad_means)
                #     valid_mask =None
                # wandb report
                test_psnr,history_data = training_report(wandb_run,test_loader,iteration, scene.model_path, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, 
                ( pipe, background) , loss_dict=loss_dict,history_data=history_data,stage=stage )
                if (iteration in testing_iterations):
                    if test_psnr >= best_psnr:
                        best_psnr = test_psnr
                        print("\n[ITER {}] Saving best checkpoint".format(iteration))
                        save_path = os.path.join(scene.model_path + "/point_cloud/chkpnt_best.pth")
                        mkdir_p(os.path.dirname(save_path))
                        torch.save((gaussians.capture(), iteration), save_path)
            
                #save
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)




                # Densification and pruning here
                if iteration < opt.densify_until_iter :
                    if opt.batch>1:
                        visibility_count = torch.stack(batch_visibility_filter,1).sum(1) #计算batch中每个点的可见总数
                        visibility_filter = visibility_count > 0
                        radii = torch.stack(batch_radii,1).max(1)[0]
                        
                        batch_viewspace_point_grad = torch.stack(batch_point_grad,1).sum(1)#将grad加起来
                        batch_viewspace_point_grad[visibility_filter] = batch_viewspace_point_grad[visibility_filter]  / visibility_count[visibility_filter] #grad除以可见次数，得到batch平均grad
                        batch_viewspace_point_grad = batch_viewspace_point_grad.unsqueeze(1)
                        # if hasattr(gaussians,"_trbf_center"):
                        #     batch_t_grad = torch.norm(gaussians._trbf_center.grad.clone(),dim=-1).squeeze()
                        #     # print(batch_t_grad.shape)
                        #     # print(visibility_filter.shape)
                        #     batch_t_grad[visibility_filter] = batch_t_grad[visibility_filter]*opt.batch  / visibility_count[visibility_filter] #grad除以可见次数，得到batch平均grad
                        #     batch_t_grad = batch_t_grad.unsqueeze(1)
                        #     # print(batch_t_grad)
                        # else:
                        batch_t_grad = None
                    if valid_mask is not None:
                        for key_index in key_index_tuple:

                            combined_mask = torch.zeros_like(valid_mask,dtype=bool,device="cuda")
                            combined_mask[valid_mask]=visibility_filter
                            # print(gaussians.max_radii2D.shape)
                            gaussians.max_radii2D[combined_mask,key_index] = torch.max(gaussians.max_radii2D[combined_mask,key_index], radii[visibility_filter]) #更新gs在2d情况下的最大半径,这个要写成逐K的                    
                        # print("hhh")
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter,valid_mask) #增加累计梯度
                    else:
                        # print(gaussians.max_radii2D.shape,visibility_filter.shape, radii.shape)
                        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter]) #更新gs在2d情况下的最大半径,这个要写成逐K的
                        if opt.batch>1:
                            gaussians.add_densification_stats_grad(batch_viewspace_point_grad, visibility_filter,batch_t_grad) #增加累计梯度
                        else:
                            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter) #增加累计梯度
                flag = controlgaussians(opt, gaussians, densify, iteration, scene,  visibility_filter, radii, viewspace_point_tensor, flag,  traincamerawithdistance=None, maxbounds=maxbounds,minbounds=minbounds)
                #for test
                # if iteration %100 ==0:
                #     prune_mask =  (gaussians.get_opacity < opt.opthr).squeeze()
                                        
                #     gaussians.prune_points(prune_mask)
                #     torch.cuda.empty_cache()
                #     scene.recordpoints(iteration, "addionally prune_mask")
                    
                # guided sampling step。这个东西虽然只有一次，不过说不定可以用来进行场景切换，新物体重建之类的建模.暂时只在multiview的情况下考虑
                # print(iteration > emsstartfromiterations ,flagems == 2 , emscnt < selectedlength , viewpoint_cam.image_name in selectviews , (iteration - lasterems > 100))
                if opt.multiview and iteration > emsstartfromiterations and flagems == 2 and emscnt < selectedlength and viewpoint_cam.image_name in selectviews and (iteration - lasterems > 100): #["camera_0002"] :#selectviews :  #["camera_0002"]:
                    emscnt += 1
                    lasterems = iteration
                    ssimcurrent = ssim(image.detach(), gt_image.detach()).item()
                    scene.recordpoints(iteration, "ssim_" + str(ssimcurrent))
                    # some scenes' strcture is already good, no need to add more points
                    if ssimcurrent < 0.88:
                        imageadjust = image /(torch.mean(image)+0.01) # 
                        gtadjust = gt_image / (torch.mean(gt_image)+0.01)
                        diff = torch.abs(imageadjust   - gtadjust)
                        diff = torch.sum(diff,        dim=0) # h, w
                        diff_sorted, _ = torch.sort(diff.reshape(-1)) 
                        numpixels = diff.shape[0] * diff.shape[1]
                        threshold = diff_sorted[int(numpixels*opt.emsthr)].item()
                        outmask = diff > threshold#  
                        kh, kw = 16, 16 # kernel size
                        dh, dw = 16, 16 # stride
                        idealh, idealw = int(image.shape[1] / dh  + 1) * kw, int(image.shape[2] / dw + 1) * kw # compute padding  
                        outmask = torch.nn.functional.pad(outmask, (0, idealw - outmask.shape[1], 0, idealh - outmask.shape[0]), mode='constant', value=0)
                        patches = outmask.unfold(0, kh, dh).unfold(1, kw, dw)
                        dummypatch = torch.ones_like(patches)
                        patchessum = patches.sum(dim=(2,3)) 
                        patchesmusk = patchessum  >  kh * kh * 0.85
                        patchesmusk = patchesmusk.unsqueeze(2).unsqueeze(3).repeat(1,1,kh,kh).float()
                        patches = dummypatch * patchesmusk

                        depth = render_pkg["depth"]
                        depth = depth.squeeze(0)
                        idealdepthh, idealdepthw = int(depth.shape[0] / dh  + 1) * kw, int(depth.shape[1] / dw + 1) * kw # compute padding for depth

                        depth = torch.nn.functional.pad(depth, (0, idealdepthw - depth.shape[1], 0, idealdepthh - depth.shape[0]), mode='constant', value=0)

                        depthpaches = depth.unfold(0, kh, dh).unfold(1, kw, dw)
                        dummydepthpatches =  torch.ones_like(depthpaches)
                        a,b,c,d = depthpaches.shape
                        depthpaches = depthpaches.reshape(a,b,c*d)
                        mediandepthpatch = torch.median(depthpaches, dim=(2))[0]
                        depthpaches = dummydepthpatches * (mediandepthpatch.unsqueeze(2).unsqueeze(3))
                        unfold_depth_shape = dummydepthpatches.size()
                        output_depth_h = unfold_depth_shape[0] * unfold_depth_shape[2]
                        output_depth_w = unfold_depth_shape[1] * unfold_depth_shape[3]

                        patches_depth_orig = depthpaches.view(unfold_depth_shape)
                        patches_depth_orig = patches_depth_orig.permute(0, 2, 1, 3).contiguous()
                        patches_depth = patches_depth_orig.view(output_depth_h, output_depth_w).float() # 1 for error, 0 for no error

                        depth = patches_depth[:render_pkg["depth"].shape[1], :render_pkg["depth"].shape[2]]
                        depth = depth.unsqueeze(0)


                        midpatch = torch.ones_like(patches)
        

                        for i in range(0, kh,  2):
                            for j in range(0, kw, 2):
                                midpatch[:,:, i, j] = 0.0  
    
                        centerpatches = patches * midpatch

                        unfold_shape = patches.size()
                        patches_orig = patches.view(unfold_shape)
                        centerpatches_orig = centerpatches.view(unfold_shape)

                        output_h = unfold_shape[0] * unfold_shape[2]
                        output_w = unfold_shape[1] * unfold_shape[3]
                        patches_orig = patches_orig.permute(0, 2, 1, 3).contiguous()
                        centerpatches_orig = centerpatches_orig.permute(0, 2, 1, 3).contiguous()
                        centermask = centerpatches_orig.view(output_h, output_w).float() # H * W  mask, # 1 for error, 0 for no error
                        centermask = centermask[:image.shape[1], :image.shape[2]] # reverse back
                        
                        errormask = patches_orig.view(output_h, output_w).float() # H * W  mask, # 1 for error, 0 for no error
                        errormask = errormask[:image.shape[1], :image.shape[2]] # reverse back

                        H, W = centermask.shape

                        offsetH = int(H/10)
                        offsetW = int(W/10)

                        centermask[0:offsetH, :] = 0.0
                        centermask[:, 0:offsetW] = 0.0

                        centermask[-offsetH:, :] = 0.0
                        centermask[:, -offsetW:] = 0.0


                        depth = render_pkg["depth"]
                        depthmap = torch.cat((depth, depth, depth), dim=0)
                        invaliddepthmask = depth == 15.0

                        pathdir = scene.model_path + "/ems_" + str(emscnt-1)
                        if not os.path.exists(pathdir): 
                            os.makedirs(pathdir)
                        
                        depthmap = depthmap / torch.amax(depthmap)
                        invalideptmap = torch.cat((invaliddepthmask, invaliddepthmask, invaliddepthmask), dim=0).float()  


                        torchvision.utils.save_image(gt_image, os.path.join(pathdir,  "gt" + str(iteration) + ".png"))
                        torchvision.utils.save_image(image, os.path.join(pathdir,  "render" + str(iteration) + ".png"))
                        torchvision.utils.save_image(depthmap, os.path.join(pathdir,  "depth" + str(iteration) + ".png"))
                        torchvision.utils.save_image(invalideptmap, os.path.join(pathdir,  "indepth" + str(iteration) + ".png"))
                        

                        badindices = centermask.nonzero()
                        diff_sorted , _ = torch.sort(depth.reshape(-1)) 
                        N = diff_sorted.shape[0]
                        mediandepth = int(0.7 * N)
                        mediandepth = diff_sorted[mediandepth]

                        depth = torch.where(depth>mediandepth, depth,mediandepth )

                    
                        totalNnewpoints = gaussians.addgaussians(badindices, viewpoint_cam, depth, gt_image, numperay=opt.farray,ratioend=opt.rayends,  depthmax=depthdict[viewpoint_cam.image_name], shuffle=(opt.shuffleems != 0))

                        gt_image = gt_image * errormask
                        image = render_pkg["render"] * errormask

                        scene.recordpoints(iteration, "after addpointsbyuv")
                        print(gaussians._xyz.shape, gaussians.xyz_gradient_accum.shape)
                        torchvision.utils.save_image(gt_image, os.path.join(pathdir,  "maskedudgt" + str(iteration) + ".png"))
                        torchvision.utils.save_image(image, os.path.join(pathdir,  "maskedrender" + str(iteration) + ".png"))
                        visibility_filter = torch.cat((visibility_filter, torch.zeros(totalNnewpoints).cuda(0)), dim=0)
                        visibility_filter = visibility_filter.bool()
                        radii = torch.cat((radii, torch.zeros(totalNnewpoints).cuda(0)), dim=0)
                        viewspace_point_tensor = torch.cat((viewspace_point_tensor, torch.zeros(totalNnewpoints, 3).cuda(0)), dim=0)


                
                # Optimizer step
                if iteration < opt.iterations:
                    # for group in gaussians.optimizer.param_groups:
                    #     if len(group["params"]) == 1 :
                    #         # extension_tensor = tensors_dict[group["name"]]
                    #         stored_state = gaussians.optimizer.state.get(group['params'][0], None)
                    #         print(stored_state["exp_avg"].shape)
                    #         print(stored_state["exp_avg_sq"].shape)
                    gaussians.optimizer.step() #根据梯度更新参数
                    
                    gaussians.optimizer.zero_grad(set_to_none = True)

def training_report(wd_writer, test_loader,iteration, model_path, loss, l1_loss, elapsed, testing_iterations, scene : Scene,renderFunc, renderArgs,history_data=None,loss_dict=None,**renderKwargs):
    if  wd_writer:
        wandb.log({
            # 'train_loss_patches/l1_loss': Ll1.item(),
            # 'train_loss_patches/ssim_loss': Ll1.item(),
            'train_loss_patches/total_loss': loss.item(),
            'iter_time': elapsed,
            'total_points': scene.gaussians.get_xyz.shape[0],
            'scene/opacity_histogram': wandb.Histogram(scene.gaussians.get_opacity.cpu().numpy()),
        }, step=iteration)
        if hasattr(scene.gaussians,"_trbf_center"):
            wandb.log({
                'scene/trbf_scale_histogram':wandb.Histogram(scene.gaussians.get_trbfscale.cpu().numpy()),
                'scene/trbf_center_histogram':wandb.Histogram(scene.gaussians._trbf_center.cpu().numpy()),
                'scene/trf_center_mean':scene.gaussians.get_trbfcenter.mean().cpu().item(),
                'scene/trf_center_std':scene.gaussians.get_trbfcenter.std().cpu().item()
            },step=iteration)
            #观测前5帧的opacity分布
            select_mask = (scene.gaussians._trbf_center < (5/300)).squeeze()
            wandb.log({
                # 'scene/first_5_opacity_histogram':wandb.Histogram(scene.gaussians.get_opacity[select_mask].cpu().numpy()),
                'scene/first_5_points_num':select_mask.sum().item(),
            },step=iteration)

            #对比几个时间段的tgrad
            # t_grad = scene.gaussians.t_gradient_accum / scene.gaussians.denom
            # t_grad[t_grad.isnan()] = 0.0
            # t_grad_1_5 = t_grad[select_mask].mean()
            # select_mask = (torch.logical_and(scene.gaussians._trbf_center >= (100/300),scene.gaussians._trbf_center < (105/300))).squeeze()
            # t_grad_100_105= t_grad[select_mask].mean()
            # select_mask = (torch.logical_and(scene.gaussians._trbf_center >= (200/300),scene.gaussians._trbf_center < (205/300))).squeeze()
            # t_grad_200_205= t_grad[select_mask].mean()

            # wandb.log({
            #     'scene/t_grad_1_5':t_grad_1_5.item(),
            #     'scene/t_grad_100_105':t_grad_100_105.item(),
            #     'scene/t_grad_200_205':t_grad_200_205.item(),
            #     'scene/t_grad':wandb.Histogram(t_grad.cpu().numpy()),
            #     'scene/t_grad_mean':t_grad.mean().item(),
            # },step=iteration)
            # if iteration%20 ==0:
            # #观测scale和xyz_grad的关系
            #     scale = scene.gaussians.get_trbfscale.cpu().numpy()
            #     xyz_grad = scene.gaussians.xyz_gradient_accum / scene.gaussians.denom
            #     xyz_grad[xyz_grad.isnan()] = 0.0
            #     xyz_grad = xyz_grad.cpu().numpy()
            #     scale_bins = np.linspace(0.1,1,10)
            #     xyz_grad_means = [np.mean(xyz_grad[scale >= i -0.1 & scale < i]) for i in scale_bins]
            #     print(xyz_grad_means)
            #     if history_data is None:
            #         history_data = {"scale_keys":[],"xyz_grad_means":[]}
            #     elif "scale_keys" not in history_data or "xyz_grad_means" not in history_data:
            #         history_data["scale_keys"] = []
            #         history_data["xyz_grad_means"] = []
            #     history_data["scale_keys"].append(iteration)
            #     history_data["xyz_grad_means"].append(xyz_grad_means)
            #     wandb.log(
            #     {
            #         validation_configs['name'] + '/psnr_perframe': wandb.plot.line_series(
            #             xs=scale_bins,
            #             ys=history_data['xyz_grad_means'],
            #             keys=history_data['scale_keys'],
            #             title="viewgrad_perscale",
            #             xname="trbf_scale",
            #         )
            #     }
            # )


        if loss_dict is not None:
            loss_dict_wandb={}
            for loss_name in loss_dict.keys():
                loss_dict_wandb[f'train_loss_patches/'+loss_name[1:]+'_loss'] = loss_dict[loss_name].item() 
            # print(loss_dict_wandb)
            wandb.log(loss_dict_wandb,step=iteration)

            # if "Lrigid" in loss_dict:
            #     wandb.log({'train_loss_patches/rigid_loss': loss_dict['Lrigid'].item()}, step=iteration)
            # if "Ldepth" in loss_dict:
            #     wandb.log({'train_loss_patches/depth_loss': loss_dict['Ldepth'].item()}, step=iteration)
            # if "Ltv" in loss_dict:
            #     wandb.log({'train_loss_patches/tv_loss': loss_dict['Ltv'].item()}, step=iteration)
            # if "Lopa" in loss_dict:
            #     wandb.log({'train_loss_patches/opa_loss': loss_dict['Lopa'].item()}, step=iteration)
            # if "Lptsopa" in loss_dict:
            #     wandb.log({'train_loss_patches/pts_opa_loss': loss_dict['Lptsopa'].item()}, step=iteration)
            # if "Lsmooth" in loss_dict:
            #     wandb.log({'train_loss_patches/smooth_loss': loss_dict['Lsmooth'].item()}, step=iteration)
            # if "Llaplacian" in loss_dict:
            #     wandb.log({'train_loss_patches/laplacian_loss': loss_dict['Llaplacian'].item()}, step=iteration)

    psnr_test_iter = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        if history_data is None:
            history_data = {'psnr_perframe':[],"keys":[]}
        elif "psnr_perframe" not in history_data or "keys" not in history_data:
            history_data['psnr_perframe'] = []
            history_data["keys"] = []

        validation_configs = {'name': 'test', 'cameras' :scene.getTestCameras()}
        render_path = os.path.join(model_path, "test", "ours_{}".format(iteration), "renders")
        os.makedirs(render_path, exist_ok=True)
        # print(validation_configs)
        # for config in validation_configs:
        if validation_configs['cameras'] and len(validation_configs['cameras']) > 0:

            l1_test_list = []
            ssim_test_list = []
            msssim_test_list = []
            psnr_test_list=[]
            for idx,viewpoint in enumerate(tqdm(validation_configs['cameras'])):
                # viewpoint = batch_data
                # for viewpoint in batch_data:
                    # print(viewpoint.timestamp)
                    gt_image = viewpoint.original_image.float().cuda()
                    viewpoint = viewpoint.cuda()
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs,**renderKwargs )
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)                
                    # depth = easy_cmap(render_pkg['depth'][0])
                    # alpha = torch.clamp(render_pkg['alpha'], 0.0, 1.0).repeat(3,1,1)

                    #这里还是不能代替test，并不能展示全部的图像,且上传非常慢。要么就不显示图像了
                    # if wd_writer and (idx %1==0):
                    #     grid = [gt_image, image, depth]
                    #     grid = make_grid(grid, nrow=2)
                    #     wandb.log({config['name'] + "_view_{}/gt_vs_render".format(viewpoint.image_name): [wandb.Image(grid, caption="Ground Truth vs. Rendered")]}, step=iteration)

                        # tb_writer.add_images(config['name'] + "_view_{}/gt_vs_render".format(viewpoint.image_name), grid[None], global_step=iteration)
                    # if config['name'] == 'test':
                    # print(psnr(image, gt_image).mean())
                    psnr_test_list.append(psnr(image, gt_image).mean().double().item())
                    l1_test_list.append(l1_loss(image, gt_image).mean().double().item())
                    ssim_test_list.append(ssim(image, gt_image).mean().double().item())
                    msssim_test_list.append(msssim(image[None].cpu(), gt_image[None].cpu()))
                    if idx%5==0:
                        #每5张保存一次


                        torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

                    # print(psnr_test_list)
            # wandb.log({"PSNR/Iteration": psnr_test_list}, step=iteration)
            psnr_test =np.mean(psnr_test_list)
            l1_test = np.mean(l1_test_list)
            ssim_test = np.mean(ssim_test_list)
            msssim_test = np.mean(msssim_test_list)
            frame_idx_list =[i for i in range(len(psnr_test_list))]
            history_data['psnr_perframe'].append(psnr_test_list)
            history_data['keys'].append(iteration)
            # print(history_data['psnr_perframe'])
            # print(history_data['keys'])
            print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, validation_configs['name'], l1_test, psnr_test))
            if wd_writer:
                wandb.log({
                    validation_configs['name'] + '/ l1_loss': l1_test,
                    validation_configs['name'] + '/psnr': psnr_test,
                    validation_configs['name'] + '/ssim': ssim_test,
                    validation_configs['name'] + '/ msssim': msssim_test
                }, step=iteration)
                wandb.log(
                    {
                        validation_configs['name'] + '/psnr_perframe': wandb.plot.line_series(
                            xs=frame_idx_list,
                            ys=history_data['psnr_perframe'],
                            keys=history_data['keys'],
                            title="psnr_perframe",
                            xname="frames",
                        )
                    }
                )
            ##write to json
            full_dict ={}
            per_view_dict = {}
            full_dict.update({"SSIM": ssim_test.item(),
                                "PSNR": psnr_test.item(),
                                # "LPIPS": torch.tensor(lpipss).mean().item(),
                                # "ssimsv2": torch.tensor(ssimsv2).mean().item(),
                                # "LPIPSVGG": torch.tensor(lpipssvggs).mean().item(),
                                # "times": torch.tensor(times).mean().item()
                                })
        
            per_view_dict.update({"SSIM": {name: ssim for ssim, name in zip(ssim_test_list, frame_idx_list)},
                                                                    "PSNR": {name: psnr for psnr, name in zip(psnr_test_list, frame_idx_list)},
                                                                    # "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                                    # "ssimsv2": {name: v for v, name in zip(torch.tensor(ssimsv2).tolist(), image_names)},
                                                                    # "LPIPSVGG": {name: lpipssvgg for lpipssvgg, name in zip(torch.tensor(lpipssvggs).tolist(), image_names)},
                                                                    })
        
            
            
            with open(model_path + "/" + str(iteration) + "_runtimeresults.json", 'w') as fp:
                    json.dump(full_dict, fp, indent=True)

            with open(model_path + "/" + str(iteration) + "_runtimeperview.json", 'w') as fp:
                json.dump(per_view_dict, fp, indent=True)
            if validation_configs['name'] == 'test':
                psnr_test_iter = psnr_test.item()
                    
    torch.cuda.empty_cache()
    return psnr_test_iter,history_data
if __name__ == "__main__":
    # print(123)
    print("current pid:",os.getpid())
    args, lp_extract, op_extract, pp_extract = getparser()
    print("start_train")
    if args.model_path == "":
        args.model_path = os.path.join("log",os.path.join(args.dataset, args.exp_name ))
    print(args.model_path)
    wandb_run = None
    if not args.no_wandb:
        tags = ['test']
        wandb_run = wandb.init(project=args.dataset, name=args.exp_name,config=args,save_code=True,resume=False,tags=tags) #resume为true并没有什么好处
    try:
        train(lp_extract, op_extract, pp_extract, args.save_iterations,args.testing_iterations, args.debug_from, checkpoint=args.checkpoint,densify=args.densify, duration=args.duration, wandb_run=wandb_run,rgbfunction=args.rgbfunction, rdpip=args.rdpip)
    except Exception as e:
        print("Error during training: ", e)
        traceback.print_exc()
        wandb.finish()
        raise e
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        wandb.finish()
    # All done
    finally:
        # print("\nTraining complete.")
        if wandb_run:
            wandb.finish()
