from torch.utils.data import Dataset,DataLoader
from scene.cameras import Camera
import numpy as np
import random 

from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCamv2
from utils.graphics_utils import focal2fov
class GSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        # dataset_type,
        resolution_scale
    ):
        self.dataset = dataset
        self.args = args
        # self.dataset_type=dataset_type
        self.resolution_scale = resolution_scale
    def __getitem__(self, index):
        # breakpoint()
        # print("getitem")
        return loadCamv2(self.args, index, self.dataset[index], self.resolution_scale)
        # if self.dataset_type != "PanopticSports":
        #     try:
        #         image, w2c, time = self.dataset[index]
        #         R,T = w2c
        #         FovX = focal2fov(self.dataset.focal[0], image.shape[2])
        #         FovY = focal2fov(self.dataset.focal[0], image.shape[1])
        #         mask=None
        #     except:
        #         # caminfo = self.dataset[index]
        #         # image = caminfo.image
        #         # R = caminfo.R
        #         # T = caminfo.T
        #         # FovX = caminfo.FovX
        #         # FovY = caminfo.FovY
        #         # time = caminfo.time
    
        #         # mask = caminfo.mask
        #     # return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
        #     #                   image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time,
        #     #                   mask=mask)
        # else:
        #     return self.dataset[index]
    def __len__(self):
        
        return len(self.dataset)


class IdxDataset(Dataset):
    '''通过idx构建的dataset'''
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class TimeBatchDataset(Dataset):
    '''通过idx构建的dataset'''
    def __init__(self, data,key_frame_dict,batch_size,scene):
        self.data = data
        self.key_frame_dict = key_frame_dict
        self.batch_size = batch_size
        self.scene = scene
    def __len__(self):
        return 30000

    def __getitem__(self, index):
        # print(index)
        #方案1
        # key = index%len(self.key_frame_dict.keys())
        # frame_list = self.key_frame_dict[key]
        # frame_idx = random.sample(frame_list,1)[0]
        # cam_idx_list = self.data[frame_idx]
        # cam_idx = random.sample(cam_idx_list,1)[0]
        # return self.scene.getTrainCameras()[cam_idx]

        #方案2
        key_list = list(self.key_frame_dict.keys())
        key_pick_list = random.sample(key_list,self.batch_size)
        real_batch = []
        for key in key_pick_list:
            frame_list = self.key_frame_dict[key]
            frame_idx = random.sample(frame_list,1)[0]
            cam_idx_list = self.data[frame_idx]
            cam_idx = random.sample(cam_idx_list,1)[0]
            real_batch.append(self.scene.getTrainCameras()[cam_idx])
        return real_batch

class SameTimeDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, scene,shuffle=True):
        super().__init__(dataset, batch_size=1, shuffle=shuffle, num_workers=2,collate_fn=self.collate_fn)
        self.real_batchsize = batch_size
        self.scene = scene
    def collate_fn(self, batch):
        # 找到一个 batch 中数据来自的时间
        # 从选定的时间中获取所有样本
        assert len(batch) == 1
        idx_same_time = random.sample(batch[0],self.real_batchsize)
        batch_same_time = [self.scene.getTrainCameras()[idx] for idx in idx_same_time ]
        # print("collect")
        return batch_same_time #直接返回list

class KeyIndexDataLoader(DataLoader):
    def __init__(self, dataset, key_frame_dict,traincameralist,batch_size, scene,shuffle=True):
        # print(traincameralist)
        # print(key_index_dict)
        # key_index_list = list(key_index_dict.keys())
        # total_frame = list(dataset.data.keys()) #300帧
        # print(total_frame)
        #构建key和frame的关系
        # key_frame_dict = {}
        # for idx,key in enumerate(key_index_list[:-1]):
        #     key_frame_dict[key] = [frame for frame in total_frame if traincameralist[dataset[frame][0]].timestamp >=key_index_dict[key] and traincameralist[dataset[frame][0]].timestamp< key_index_dict[key+1]]
        #     if idx == len(key_index_list)-2:
        #         key_frame_dict[key]+=[frame for frame in total_frame if traincameralist[frame].timestamp ==key_index_dict[key+1]]
        # for k,v in key_frame_dict.items():
        #     print(k,v)
        # batch_size = min(batch_size,len(key_frame_dict.keys()))
        super().__init__(key_frame_dict, batch_size=batch_size, shuffle=shuffle, num_workers=2,collate_fn=self.collate_fn)
        self.real_batchsize = batch_size
        self.scene = scene
        self._dataset = dataset
    def collate_fn(self, batch):
        # 找到一个 batch 中数据来自的时间
        # 从选定的时间中获取所有样本
        # assert len(batch) == 1
        print(batch)
        real_batch = []
        for frame_list in batch:
            frame_idx = random.sample(frame_list,1)[0]
            print(frame_idx)
            cam_idx_list = self._dataset[frame_idx]
            cam_idx = random.sample(cam_idx_list,1)[0]
            print(cam_idx)
            real_batch.append(self.scene.getTrainCameras()[cam_idx])
        print(real_batch)
        # exit()
        # print("collect")
        return real_batch #直接返回list

class randomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size,scene, shuffle=True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
        self.scene = scene

    def collate_fn(self, batch):
        # 找到一个 batch 中数据来自的时间
        # 从选定的时间中获取所有样本
        real_batch = [self.scene.getTrainCameras()[idx] for idx in batch ]

        return torch.stack(batch_same_time)
