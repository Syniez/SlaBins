from PIL import ImageFile
from PIL import Image
import pandas as pd
import numpy as np
import argparse
import shutil
import json
import cv2
import os
import sys
import glob
import copy
import matplotlib.cm
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import struct
import pickle


from loadCalibration import loadCalibrationCameraToPose, loadCalibrationRigid
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
from collections import Counter
from tqdm import tqdm
from skimage.morphology import erosion, dilation, opening, closing, white_tophat, disk
from project import *


class Kitti360Viewer3DRaw(object):
    def __init__(self, path=None, seq=0, mode='velodyne'):
        kitti360Path = path
        sequence = '2013_05_28_drive_%04d_sync' % seq
        self.raw3DPcdPath  = os.path.join(kitti360Path, 'data_3d_raw', sequence, 'velodyne_points', 'data')
        #self.raw3DPcdPath  = os.path.join(kitti360Path, 'data_3d_raw', sequence, self.sensor_dir, 'data')

    def loadVelodyneData(self, frame=0):
        pcdFile = os.path.join(self.raw3DPcdPath, '%010d.bin' % frame)
        if not os.path.isfile(pcdFile):
            raise RuntimeError('%s does not exist!' % pcdFile)
        pcd = np.fromfile(pcdFile, dtype=np.float32)
        pcd = np.reshape(pcd,[-1,4])
        return pcd 

    def loadSickData(self, frame=0):
        pcdFile = os.path.join(self.raw3DPcdPath, '%010d.bin' % frame)
        if not os.path.isfile(pcdFile):
            raise RuntimeError('%s does not exist!' % pcdFile)
        pcd = np.fromfile(pcdFile, dtype=np.float32)
        pcd = np.reshape(pcd,[-1,2])
        pcd = np.concatenate([np.zeros_like(pcd[:,0:1]), -pcd[:,0:1], pcd[:,1:2]], axis=1)
        return pcd 


def create_fisheye_raw_point_cloud(rgb, norm, angle_lut, theta_lut):
    rgb = list(map(lambda x: x.reshape(x.shape[1]*x.shape[0], 1), cv2.split(rgb)))

    def _img2world(norm, angle_of_incidence, theta):
        
        norm = norm.reshape((norm.shape[0]*norm.shape[1], -1))
        r_world = norm * np.sin(angle_of_incidence)
        x = r_world * np.cos(theta)
        y = r_world * np.sin(theta)
        # Obtain `z` from the norm
        z = np.cos(angle_of_incidence) * norm
        return x, y, z

    x, y, z = _img2world(norm, angle_lut, theta_lut)
    point_cloud = np.hstack([x, y, z, rgb[0], rgb[1], rgb[2]])
    return point_cloud


def sum_velo(velo,cam0toworld,cam0tovelo,frame,interval):
    summed_velo = velo.loadVelodyneData(frame)
    summed_velo[:,3] = 1
    if frame < interval:
        for i in range(1,interval+1):
            plus_velo = velo.loadVelodyneData(frame+i)
            plus_velo[:,3] = 1
            v_proj__ = cam0tovelo@np.linalg.inv(cam0toworld[str(frame)])@cam0toworld[str(frame+i)]@np.linalg.inv(cam0tovelo)@plus_velo.T
            summed_velo = np.concatenate([summed_velo,v_proj__.T],axis=0)
    elif frame > len(cam0toworld)-interval:
        for i in range(1,interval+1):
            minus_velo = velo.loadVelodyneData(frame-i)
            minus_velo[:,3] = 1
            v_proj_ = cam0tovelo@np.linalg.inv(cam0toworld[str(frame)])@cam0toworld[str(frame-i)]@np.linalg.inv(cam0tovelo)@minus_velo.T
            summed_velo = np.concatenate([summed_velo,v_proj_.T],axis=0)
    else:
        for i in range(1,interval+1):
            plus_velo = velo.loadVelodyneData(frame+i)
            plus_velo[:,3] = 1
            minus_velo = velo.loadVelodyneData(frame-i)
            minus_velo[:,3] = 1
            v_proj_ = cam0tovelo@np.linalg.inv(cam0toworld[str(frame)])@cam0toworld[str(frame-i)]@np.linalg.inv(cam0tovelo)@minus_velo.T
            v_proj__ = cam0tovelo@np.linalg.inv(cam0toworld[str(frame)])@cam0toworld[str(frame+i)]@np.linalg.inv(cam0tovelo)@plus_velo.T
            summed_velo = np.concatenate([summed_velo,v_proj_.T],axis=0)
            summed_velo = np.concatenate([summed_velo,v_proj__.T],axis=0)
        
    return summed_velo


def dense_depth_projection(path=None, cam_id=0, seq=0, frame=0):
    kitti360Path =path
    sequence = '2013_05_28_drive_%04d_sync'%seq

    camera = CameraFisheye(kitti360Path, sequence, cam_id)
    velo = Kitti360Viewer3DRaw(path=kitti360Path, mode='velodyne', seq=seq)
    
    with open(f'./make_list_txt/cam02world_seq{seq}.pkl','rb') as z:
        cam02world_dict = pickle.load(z)
    
    # cam_0 to velo
    fileCameraToVelo = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_velo.txt')
    TrCam0ToVelo = loadCalibrationRigid(fileCameraToVelo)
    b=TrCam0ToVelo
    # all cameras to system center 
    fileCameraToPose = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_pose.txt')
    TrCamToPose = loadCalibrationCameraToPose(fileCameraToPose)

    # velodyne to all cameras
    TrVeloToCam = {}
    for k, v in TrCamToPose.items():
        # Tr(cam_k -> velo) = Tr(cam_k -> cam_0) @ Tr(cam_0 -> velo)
        TrCamkToCam0 = np.linalg.inv(TrCamToPose['image_00']) @ TrCamToPose[k]
        TrCamToVelo = TrCam0ToVelo @ TrCamkToCam0
        # Tr(velo -> cam_k)
        TrVeloToCam[k] = np.linalg.inv(TrCamToVelo)
    
    else:
        TrVeloToRect = TrVeloToCam['image_%02d' % cam_id]


    points = sum_velo(velo, cam02world_dict, b, frame, 5)
    # transfrom velodyne points to camera coordinate
    pointsCam = np.matmul(TrVeloToRect, points.T).T
    pointsCam = pointsCam[:,:3]
    # project to image space
    u,v, depth= camera.cam2image(pointsCam.T)
    u = u.astype(np.int)
    v = v.astype(np.int)

    # prepare depth map for visualization
    depthMap = np.zeros((camera.height, camera.width))
    #depthImage = np.zeros((camera.height, camera.width, 3))
    mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<camera.width), v>=0), v<camera.height)
    mask = np.logical_and(np.logical_and(mask, depth>0), depth<40) # for omnidet
    depthMap[v[mask],u[mask]] = depth[mask]
    #layout = (2,1) if cam_id in [0,1] else (1,2)
    
    depth = depth.copy()
    masked_u = u[mask]
    masked_v = v[mask]
    masked_depth = depth[mask]
    zeros_depthMap = np.zeros((camera.height, camera.width)) 
    
    distance_layer_source = np.ceil(masked_depth)
    distance_layer_source = distance_layer_source.astype(np.int64)

    #counter
    keys_list =Counter(distance_layer_source).keys()
    keys_list =list(keys_list)

    target_list = []
    for i in keys_list:
        indx = np.where(distance_layer_source==i)
        zeros_depthMap_ =zeros_depthMap.copy()
        zeros_depthMap_[masked_v[indx],masked_u[indx]]=i
    
        temp_im = Image.fromarray(zeros_depthMap_.astype(np.uint8))
        target_image=dilation(temp_im,disk(3))
        target_list.append(target_image)
    target_array = np.array(target_list)
    #for min_layer computation
    flat_target_array = target_array.reshape(-1).copy()
    flat_target_array[np.where(flat_target_array==0)[0]]=255
    target_array2=flat_target_array.reshape(-1, 1400, 1400)

    layer_array = np.min(target_array2,axis=0)
    flat_layer_array = layer_array.reshape(-1).copy()
    flat_layer_array[np.where(flat_layer_array==255)[0]] = 0
    layer_array_image = flat_layer_array.reshape(1400,1400)

    correction_map = depthMap.copy()
    correction_map[depthMap>layer_array_image] =0            
    
    return correction_map


def warp(ori_rgb, ori_depth, intrinsics, point_cloud, angle_diff, width, height):

    # intrinsics
    aspect_ratio = intrinsics['aspect_ratio']
    k1 = intrinsics['k1']
    k2 = intrinsics['k2']
    k3 = intrinsics['k3']
    k4 = intrinsics['k4']
    cx = intrinsics['cx_offset']
    cy = intrinsics['cy_offset']

    # vacant array
    warp_rgb = np.zeros_like(ori_rgb)
    warp_depth = np.zeros_like(ori_depth)

    # rotation
    angle_diff = np.radians(angle_diff)
    R = np.array([[1,0,0],[0,np.cos(-angle_diff), -np.sin(-angle_diff)], [0, np.sin(-angle_diff), np.cos(-angle_diff)]])
    rotated_point = (R @ point_cloud[:, :3].T).T
    rotated_point = np.concatenate((rotated_point, point_cloud[:, 3:6], point_cloud[:, 6:]), 1)
    #rotated_point = np.concatenate((rotated_point, point_cloud[:, 3:]), 1)

    # center point
    px = cx + width/2 - 0.5
    py = cy + height/2 - 0.5

    # projection
    #rotated_point = rotated_point.reshape((height, width, 9))
    rotated_point = rotated_point.reshape((height, width, 7))
    chi = np.sqrt(rotated_point[:,:,0] **2 + rotated_point[:,:,1] **2)
    theta = np.pi/2 - np.arctan2(rotated_point[:,:,2], chi)
    rho = k1 * theta + k2 * theta**2 + k3 * theta**3 + k4 * theta**4
    
    x = rho * rotated_point[:,:,0] / (chi+1e-6) + px
    y = rho * rotated_point[:,:,1] / (chi+1e-6) * aspect_ratio + py

    x_mask = np.logical_and(x>=0, x<width)
    y_mask = np.logical_and(y>=0, y<height)
    
    x = (x * x_mask)#.reshape(width * height, -1)
    y = (y * y_mask)#.reshape(width * height, -1)
    z = rotated_point[:,:,2]
    corner_pixel = z[0,0]
    z_mask = z.copy()

    for i in range(width):
        for j in range(height):
            if z_mask[j,i] == corner_pixel:
                z_mask[j,i] = 0
            else:
                z_mask[j,i] = 1
            if int(x[j,i]) == 0 or int(y[j,i]) == 0:
               continue
            if int(x[j,i]) >= width or int(y[j,i]) >= height:
                continue
            
            warp_rgb[j,i] = [rotated_point[int(y[j,i]), int(x[j,i]), 5], rotated_point[int(y[j,i]), int(x[j,i]), 4], rotated_point[int(y[j,i]), int(x[j,i]), 3]]
            warp_depth[j,i] = np.linalg.norm(rotated_point[int(y[j,i]), int(x[j,i]), :3])

    z_mask = np.transpose(np.tile(z_mask, (3, 1, 1)), (1, 2, 0))
    warp_rgb_ = warp_rgb * z_mask
        
    return warp_rgb_, warp_depth


def main(args):
    kittipath = '/KITTI360/'
    rgb_dst_path = ''
    dep_dst_path = ''

    cam = (args.cam).zfill(2)
    angles = [0, 20, 40, 60]

    luts = pd.read_pickle("./data/KITTI.pkl")
    keys = list(luts.keys())
    
    theta_l = luts[keys[0]]['theta']
    angle_l = luts[keys[0]]['angle_maps']
    theta_r = luts[keys[1]]['theta']
    angle_r = luts[keys[1]]['angle_maps']

    with open('./data/MVL_KITTI.json', 'r') as l:
        intrinsics_l = json.load(l)['intrinsic']   
    with open('./data/MVR_KITTI.json', 'r') as r:
        intrinsics_r = json.load(r)['intrinsic']   

    with open (args.txt) as f:
        seqs = f.readlines()
    
    for i in tqdm(seqs):
        #if args.seq == (i.split()[1][17:21]).lstrip("0"):
        if args.seq == (i.split()[1][17:21])[-1]:    
            if cam == '02':
                angle = angle_l
                theta = theta_l
                intrinsics = intrinsics_l
            else:
                angle = angle_r
                theta = theta_r
                intrinsics = intrinsics_r

            for slanted_angle in angles:
                idx = i.split()[0][-10:].lstrip("0")
                seq = i.split()[0][-15:-11][-1]
                img_path = kittipath + 'data_2d_raw/' + i.split()[1][:26] + '/image_' + cam + '/data_rgb/'+ i.split()[0][-10:] + '.png'
                camera = CameraFisheye(kittipath, i.split()[1][:26], int(cam.lstrip("0")))

                rgb = np.array(Image.open(img_path))
                dense_depth = dense_depth_projection(path=kittipath, cam_id=int(cam.lstrip("0")), seq=int(seq), frame=int(idx))
                
                dense_depth[dense_depth==0] = 60
                pcl = create_fisheye_raw_point_cloud(rgb, dense_depth, angle, theta)
                w_rgb, w_dep = warp(rgb, dense_depth, intrinsics, pcl, slanted_angle, 1400, 1400)

                img_name = (img_path.replace('/', '__').replace('.png', ''))[15:]
                w_dep[w_dep > 40] = 0
        
                cv2.imwrite(rgb_dst_path + img_name + '_rgb_' + str(90-slanted_angle) + '.png', w_rgb)
                np.save(dep_dst_path + img_name + '_dep_' + str(90-slanted_angle) + '.npy', w_dep)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt', required=True)
    parser.add_argument('-cam', required=True)
    parser.add_argument('-seq', required=True)
    args = parser.parse_args()

    main(args)
