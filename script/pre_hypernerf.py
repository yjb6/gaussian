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
import cv2 
import glob 
import tqdm 
import numpy as np 
import shutil
import pickle
import sys 
import argparse
import json
sys.path.append(".")
from thirdparty.gaussian_splatting.utils.my_utils import posetow2c_matrcs, rotmat2qvec
from thirdparty.colmap.pre_colmap import *
from thirdparty.gaussian_splatting.helper3dg import getcolmapsinglen3d

print("import ok")


def extractframes(videopath):
    cam = cv2.VideoCapture(videopath)
    ctr = 0
    sucess = True
    for i in range(300):
        if os.path.exists(os.path.join(videopath.replace(".mp4", ""), str(i) + ".png")):
            ctr += 1
    if ctr == 300 or ctr == 150: # 150 for 04_truck 
        print("already extracted all the frames, skip extracting")
        return
    ctr = 0
    while ctr < 300:
        try:
            _, frame = cam.read()
            # print(frame)
            # print(videopath)
            savepath = os.path.join(videopath.replace(".mp4", ""), str(ctr) + ".png")
            if not os.path.exists(videopath.replace(".mp4", "")) :
                os.makedirs(videopath.replace(".mp4", ""))

            cv2.imwrite(savepath, frame)
            ctr += 1 
        except:
            sucess = False
            print("error")
    cam.release()
    return


# def preparecolmapdynerf(folder, offset=0):
#     #把每个相机下的第offset帧都放到了colmap_offset下
#     folderlist = glob.glob(folder + "cam**/")
#     imagelist = []
#     savedir = os.path.join(folder, "colmap_" + str(offset))
#     if not os.path.exists(savedir):
#         os.mkdir(savedir)
#     savedir = os.path.join(savedir, "input")
#     if not os.path.exists(savedir):
#         os.mkdir(savedir)
#     for folder in folderlist :
#         imagepath = os.path.join(folder, str(offset) + ".png")
#         imagesavepath = os.path.join(savedir, folder.split("/")[-2] + ".png")

#         shutil.copy(imagepath, imagesavepath)

def preparecolmaphypernerf(folder, cam):
    '''cam格式是left/right_xxxxx'''
    #把每张照片单独保存在colmap_cam文件夹下
    # folderlist = glob.glob(folder + "cam**/")
    # camera_list=glob.glob(folder + "camera/")
    # print(folder)
    image_dir = os.path.join(folder,"rgb","2x")
    image_path = cam+".png"
    # imagelist = []
    savedir = os.path.join(folder, "colmap_" + cam)
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    savedir = os.path.join(savedir, "input")
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    imagesavepath = os.path.join(savedir, image_path)
    shutil.copy(os.path.join(image_dir,image_path), imagesavepath)
    # for image in imagelist :
    #     imagepath = os.path.join(folder, str(offset) + ".png")
    #     imagesavepath = os.path.join(savedir, folder.split("/")[-2] + ".png")

    #     shutil.copy(imagepath, imagesavepath)
    
def converthypernerftocolmapdb(path, cam):
    #将位姿信息转换成colmap格式存储
    # originnumpy = os.path.join(path, "poses_bounds.npy")
    # video_paths = sorted(glob.glob(os.path.join(path, 'cam*.mp4')))
    camera_dir=os.path.join(path,"camera")
    camera_path = os.path.join(camera_dir,cam+".json")
    # cameras.sort()
    # cams = []
    # cam = 

    projectfolder = os.path.join(path, "colmap_" + cam)
    #sparsefolder = os.path.join(projectfolder, "sparse/0")
    manualfolder = os.path.join(projectfolder, "manual")

    # if not os.path.exists(sparsefolder):
    #     os.makedirs(sparsefolder)
    if not os.path.exists(manualfolder):
        os.makedirs(manualfolder)

    savetxt = os.path.join(manualfolder, "images.txt")
    savecamera = os.path.join(manualfolder, "cameras.txt")
    savepoints = os.path.join(manualfolder, "points3D.txt")
    imagetxtlist = []
    cameratxtlist = []
    if os.path.exists(os.path.join(projectfolder, "input.db")):
        os.remove(os.path.join(projectfolder, "input.db"))

    db = COLMAPDatabase.connect(os.path.join(projectfolder, "input.db"))

    db.create_tables()

    # for jsonfile in tqdm(cameras):
    with open (os.path.join(camera_path)) as f:
        camera = json.load(f)
            # cams.append(json.load(f))
    R = np.array(camera['orientation']).T
    T = -np.array(camera['position'])@R 
    image_size = camera['image_size']

    colmapQ = rotmat2qvec(R.T)
    qevc = [str(i) for i in rotmat2qvec(R.T)]
        # T = [str(i) for i in T]
    imageid = str(1)
    cameraid = imageid
    pngname = cam + ".png"
    line = imageid+" "+" ".join(qevc)+" "+" ".join([str(i) for i in T]) +" "+ cameraid + " " + pngname + "\n"
    empltyline = "\n"
    imagetxtlist.append(line)
    imagetxtlist.append(empltyline)

    # focolength = focal
    width, height, params =  image_size[0]/2, image_size[1]/2, np.array((camera['focal_length']/2, camera['principal_point'][0]/2,camera['principal_point'][1]/2,))

    camera_id = db.add_camera(0, width, height, params)
    cameraline = str(1) + " " + "SIMPLE_PINHOLE " + str(width) +  " " + str(height) + " " + str(camera['focal_length']/2) + " " + str(camera['principal_point'][0]/2) + " " + str(camera['principal_point'][1]/2) + "\n"
    cameratxtlist.append(cameraline)

    image_id = db.add_image(pngname, camera_id,  prior_q=np.array((colmapQ[0], colmapQ[1], colmapQ[2], colmapQ[3])), prior_t=np.array((T[0], T[1], T[2])), image_id=1)
    db.commit()
    db.close()
    # for idx in range(len(cams)):
    #     R = np.array(cam['orientation']).T
    #     T = -np.array(cam['position'])@R 
    #     image_size = cams[i]['image_size']

    #     colmapQ = rotmat2qvec(R.T)
    #     qevc = [str(i) for i in rotmat2qvec(R.T)]
    #     # T = [str(i) for i in T]
    #     imageid = str(idx+1)
    #     cameraid = imageid
    #     pngname = cam + ".png"
    #     line = imageid+" "+" ".join(qevc)+" "+" ".join([str(i) for i in T]) +" "+ cameraid + " " + pngname + "\n"
    #     empltyline = "\n"
    #     imagetxtlist.append(line)
    #     imagetxtlist.append(empltyline)

    #     focolength = focal
    #     model, width, height, params = idx, image_size[0]/2, image_size[1]/2, np.array((cam['focal_length']/2, cam['principal_point'][0]/2,cam['principal_point'][1]/2,))

    #     camera_id = db.add_camera(0, width, height, params)
    #     cameraline = str(idx+1) + " " + "SIMPLE_PINHOLE " + str(width) +  " " + str(height) + " " + str(cam['focal_length']/2) + " " + str(cam['principal_point'][0]/2) + " " + str(cam['principal_point'][1]/2) + "\n"
    #     cameratxtlist.append(cameraline)

    #     image_id = db.add_image(pngname, camera_id,  prior_q=np.array((colmapQ[0], colmapQ[1], colmapQ[2], colmapQ[3])), prior_t=np.array((T[0], T[1], T[2])), image_id=i+1)
    #     db.commit()
    # with open(originnumpy, 'rb') as numpy_file:
    #     # poses_bounds = np.load(numpy_file)

    #     poses = poses_bounds[:, :15].reshape(-1, 3, 5)

    #     llffposes = poses.copy().transpose(1,2,0)
    #     w2c_matriclist = posetow2c_matrcs(llffposes)
    #     assert (type(w2c_matriclist) == list)


    #     for i in range(len(poses)):
    #         cameraname = os.path.basename(video_paths[i])[:-4]#"cam" + str(i).zfill(2)
    #         m = w2c_matriclist[i]
    #         colmapR = m[:3, :3]
    #         T = m[:3, 3]
            
    #         H, W, focal = poses[i, :, -1]
            
    #         colmapQ = rotmat2qvec(colmapR)
    #         # colmapRcheck = qvec2rotmat(colmapQ)

    #         imageid = str(i+1)
    #         cameraid = imageid
    #         pngname = cameraname + ".png"
            
    #         line =  imageid + " "

    #         for j in range(4):
    #             line += str(colmapQ[j]) + " "
    #         for j in range(3):
    #             line += str(T[j]) + " "
    #         line = line  + cameraid + " " + pngname + "\n"
    #         empltyline = "\n"
    #         imagetxtlist.append(line)
    #         imagetxtlist.append(empltyline)

    #         focolength = focal
    #         model, width, height, params = i, W, H, np.array((focolength, W//2, H//2,))

    #         camera_id = db.add_camera(1, width, height, params)
    #         cameraline = str(i+1) + " " + "PINHOLE " + str(width) +  " " + str(height) + " " + str(focolength) + " " + str(focolength) + " " + str(W//2) + " " + str(H//2) + "\n"
    #         cameratxtlist.append(cameraline)
            
    #         image_id = db.add_image(pngname, camera_id,  prior_q=np.array((colmapQ[0], colmapQ[1], colmapQ[2], colmapQ[3])), prior_t=np.array((T[0], T[1], T[2])), image_id=i+1)
    #         db.commit()
        # db.close()


    with open(savetxt, "w") as f:
        for line in imagetxtlist :
            f.write(line)
    with open(savecamera, "w") as f:
        for line in cameratxtlist :
            f.write(line)
    with open(savepoints, "w") as f:
        pass 





if __name__ == "__main__" :
    '''colmap初始化最少需要2帧，所以单目数据不能每一帧建个点云，考虑采取4dgs的做法对所有帧建一个点云。然后测试随机初始化时间和将时间统一设置在第一帧这几种做法'''
    # print("123")
    parser = argparse.ArgumentParser()
 
    parser.add_argument("--path", default="", type=str)
    # parser.add_argument("--startframe", default=0, type=int)
    # parser.add_argument("--endframe", default=50, type=int)

    args = parser.parse_args()
    path = args.path

    # startframe = args.startframe
    # endframe = args.endframe


    # if startframe >= endframe:
    #     print("start frame must smaller than end frame")
    #     quit()
    # if startframe < 0 or endframe > 300:
    #     print("frame must in range 0-300")
    #     quit()
    if not os.path.exists(path):
        print("path not exist")
        quit()
    
    if not path.endswith("/"):
        path = path + "/"
    
    
    camera_list = [camera.split('.')[-2] for camera in os.listdir(os.path.join(path,"camera"))]
    camera_list.sort()
    print(camera_list)

    # # ## step2 prepare colmap input 
    print("start preparing colmap image input")
    for camera in camera_list:
        preparecolmaphypernerf(path,camera)
    
    # for offset in range(startframe, endframe):
    #     preparecolmapdynerf(videopath, offset)


    print("start preparing colmap database input")
    # # ## step 3 prepare colmap db file 
    for camera in camera_list:
        converthypernerftocolmapdb(path,camera)
    # for offset in range(startframe, endframe):
    #     convertdynerftocolmapdb(videopath, offset)


    # ## step 4 run colmap, per frame, if error, reinstall opencv-headless 
    for camera in camera_list:
        getcolmapsinglen3d(path, camera)




