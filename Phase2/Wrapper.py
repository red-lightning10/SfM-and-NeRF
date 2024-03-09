import argparse
import glob
from tqdm import tqdm
import random
# from torch.utils.tensorboard import SummaryWriter
import imageio
import torch
import matplotlib.pyplot as plt
import os
import json
import cv2
import numpy as np
from NeRFModel import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
print(device)
def loadDataset(data_path, mode):
    """
    Input:
        data_path: dataset path
        mode: train or test or val
    Outputs:
        camera_info: image width, height, camera matrix 
        images: images
        pose: corresponding camera pose in world frame
    """
    json_file_path = os.path.join(data_path, "transforms_" + mode)
    json_file = glob.glob(json_file_path + ".json")
    image_paths = []
    with open(json_file[0]) as f:
        data = json.load(f)
        camera_fov = float(data['camera_angle_x'])
        print(camera_fov)
        pose = data['frames']
        print(pose)
        for i in range(len(pose)):
            pose[i]['transform_matrix'] = torch.tensor(pose[i]['transform_matrix'])
            pose[i]['rotation'] = torch.tensor(pose[i]['rotation'])
            pose[i]['file_path'] = os.path.join(data_path, pose[i]['file_path'][2:])
            image_paths.append(pose[i]['file_path'] + ".png")
    images = []
    for i in range(len(image_paths)):
        img = cv2.imread(image_paths[i])
        images.append(torch.tensor(img))

    camera_info = { "H" : images[0].shape[0], "W" : images[0].shape[1], \
                   "f" : torch.tensor(0.5*images[0].shape[1]/np.tan(0.5*camera_fov)).to(device)}
    return torch.stack(images).to(device), pose, camera_info


def PixelToRay(camera_info, pose):
    """
    Input:
        camera_info: image width, height, camera matrix 
        pose: camera pose in world frame..
    Outputs:
        ray origin and direction
    """
    width = camera_info['W']
    height = camera_info['H']
    focal_len = camera_info['f']
    grid_x, grid_y = torch.meshgrid(torch.arange(width).to(device),torch.arange(height).to(device))
    grid_x= grid_x.transpose(-1,-2)
    grid_y = grid_y.transpose(-1,-2)
    directions=torch.stack([(grid_x - width * .5) / focal_len, -(grid_y - height * .5) / focal_len, -torch.ones_like(grid_x)], dim=-1)
    rays_dir=torch.sum(directions[..., None, :] *pose[:3, :3], dim=-1)
    rays_ori= pose[:3,-1].expand(rays_dir.shape)
    return rays_dir,rays_ori

def ray_sample(ray_ori,ray_dir, near, far, sample):
    """
    Input:
        ray_ori: ray origin
        ray_dir: ray direction
        near: near range
        far: far range
        sample: sample rate
    Outputs:
        sampled rays
        len of the ray
    """
    len = torch.linspace(near,far,sample).to(device)
    # print(depth_values.shape)
    # print('------------DEPTH VALUES BEFORE NOISE----------------------------')
    noise_shape = list(ray_ori.shape[:-1])+[sample]
    # print(noise_shape)
    len = len + torch.rand(noise_shape).to(ray_ori)*(far-near)/sample
    # print(depth_values.shape)
    sample_pts = ray_ori[..., None, :] + ray_dir[..., None, :]*len[..., :, None]
    # pry()
    return sample_pts, len
    #len = np.linspace(near, far, sample)
    #pt_shape = list(ray_ori.shape[:-1])+[sample]
    #randomness =torch.rand(pt_shape)*((far-near)/sample)
    #print(np.shape(len))
    #print(np.shape(randomness))
    ##len = len + randomness
    #ample_pos = torch.tensor(ray_ori + len[:, None] * ray_dir)
    #return sample_pos, len

def render(depth, rays_ori, rgb, sigma):
    e_10 = torch.tensor([1e10], dtype = rays_ori.dtype, device = rays_ori.device)
    e_10 = e_10.expand(depth[...,:1].shape)
    delta_i= depth[...,1:] - depth[...,:-1]
    adjacent_dist = torch.concat((delta_i, e_10), dim = -1)
    alpha = 1.0 - torch.exp(-1 * sigma * adjacent_dist)
    wts = alpha * cumulative_product(1.0 - alpha + 1e-10)
    rgb_map = (wts[..., None] * rgb).sum(dim = -2)
    return wts, rgb_map

def loss_fn(groundtruth, prediction):
    """
    Input:
        groundtruth: groundtruth rgb values
        prediction: predicted rgb values
    Outputs:
        loss
    """
    torch.nn.functional.mse_loss(rgb, img)
    return loss
def train_per_epoch(device, camera_info, cam_pose, near, far, sample, high_N, batch_size, model):
    rays_dir, rays_ori = PixelToRay(camera_info, cam_pose)
    rays_pos, len = ray_sample(rays_ori, rays_dir, near, far, sample)
    flatten_ray_pos = torch.Tensor(rays_pos.reshape((-1,3)))
    out = model.pos_enc(flatten_ray_pos, high_N)
    out_batch = [out[i:i + batch_size] for i in range(0, out.shape[0], batch_size)]

    model_out = []
    for b in batch:
        model_out.append((model(b)))
    radiance = torch.cat(model_out, dim=0)
    unflatten = list(query_pt.shape[:-1]) + [4]
    radiance_field = torch.reshape(radiance, unflatten)
    sigma = torch.relu(radiance_field[...,3])
    rgb = torch.sigmoid(radiance_field[...,:3])
    wts, rgb_map = render(depth, rays_ori, rgb, sigma)
    return rgb_map
def train(images, poses, camera_info,args):
    num_epoch = args.max_iters
    sample = args.n_sample
    batch_size = args.n_rays_batch
    near = args.near
    far = args.far
    high_N = args.encode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    focal_len = camera_info['f']
    height, width = camera_info['H'], camera_info['W']

    model = NeRFmodel()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 10e-3)

    torch.manual_seed(9458)
    random.seed(9458)
    Loss = []
    Epochs = []
    for i in range(num_epoch+1):
        img_idx = random.randint(0, images.shape[0]-1)
        img = images[img_idx].to(device)
        img=img.float()
        cam_pose = poses[img_idx]['transform_matrix'].to(device)
        rgb,_,_ = train_per_epoch(device, camera_info, cam_pose, near, far, sample, high_N, batch_size, model)
        loss= loss_fn(rgb, img)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        plt.imshow(rgb.detach().cpu().numpy())
        plt.title(f"Iteration {i}")
        plt.savefig("rgb.png")
        Loss.append(loss.item())
        Epochs.append(i+1)
        plot_loss(Epochs, Loss)
    checkpoint =  './Checkpoint/' + 'model_' + str(i) + '.ckpt'
    torch.save({'epoch': Epochs, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint)
#def test(images, poses, camera_info, args):
    """
    Input:
        images: all images in dataset
        poses: corresponding camera pose in world frame
        camera_info: image width, height, camera matrix
        args: testing related parameters
    Outputs:
        rendered images
    """

def main(args):
    
    # load data
    print("Loading data...")
    images, poses, camera_info = loadDataset(args.data_path, args.mode)
    images = torch.tensor(images).to(device)
    #poses = torch.tensor(poses).to(device)
    #camera_info['f'] = camera_info['f'].to(device)
    if args.mode == 'train':
        print("Start training")
        train(images, poses, camera_info, args)
    elif args.mode == 'test':
        print("Start testing")
        args.load_checkpoint = True
        test(images, poses, camera_info, args)

def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default="C:/Users/DELL/Downloads/SfM-and-NeRF/Phase2/lego/",help="dataset path")
    parser.add_argument('--mode',default='train',help="train/test/val")
    parser.add_argument('--lrate',default=5e-4,help="training learning rate")
    parser.add_argument('--n_pos_freq',default=10,help="number of positional encoding frequencies for position")
    parser.add_argument('--n_dirc_freq',default=4,help="number of positional encoding frequencies for viewing direction")
    parser.add_argument('--n_rays_batch',default=32*32*4,help="number of rays per batch")
    parser.add_argument('--n_sample',default=32,help="number of sample per ray")
    parser.add_argument('--max_iters',default=10000,help="number of max iterations for training")
    parser.add_argument('--logs_path',default="C:/Users/DELL/Downloads/SfM-and-NeRF/Phase2/logs/",help="logs path")
    parser.add_argument('--checkpoint_path',default="C:/Users/DELL/Downloads/SfM-and-NeRF/Phase2/example_checkpoint/",help="checkpoints path")
    parser.add_argument('--load_checkpoint',default=True,help="whether to load checkpoint or not")
    parser.add_argument('--images_path', default="./image/",help="folder to store images")
    parser.add_argument('--near', type=int, default=2)
    parser.add_argument('--far',type=int, default=6)
    parser.add_argument('--encode',type=int, default=6)
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)
