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
        for i in range(len(pose)):
            pose[i]['transform_matrix'] = torch.tensor(pose[i]['transform_matrix'])
            pose[i]['rotation'] = torch.tensor(pose[i]['rotation'])
            pose[i]['file_path'] = os.path.join(data_path, pose[i]['file_path'][2:])
            image_paths.append(pose[i]['file_path'] + ".png")
    images = []
    for i in range(len(image_paths)):
        img = cv2.imread(image_paths[i])
        images.append(img)

    camera_info = { "H" : images[0].shape[0], "W" : images[0].shape[1], \
                   "f" : torch.tensor(0.5*images[0].shape[1]/np.tan(0.5*camera_fov)).to(device)}
    return images, pose, camera_info


def PixelToRay(camera_info, pose, pixelPosition, args):
    """
    Input:
        camera_info: image width, height, camera matrix 
        pose: camera pose in world frame
        pixelPoition: pixel position in the image
        args: get near and far range, sample rate ...
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

def generateBatch(images, poses, camera_info, args):
    """
    Input:
        images: all images in dataset
        poses: corresponding camera pose in world frame
        camera_info: image width, height, camera matrix
        args: get batch size related information
    Outputs:
        A set of rays
    """

def render(model, rays_origin, rays_direction, args):
    """
    Input:
        model: NeRF model
        rays_origin: origins of input rays
        rays_direction: direction of input rays
    Outputs:
        rgb values of input rays
    """

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
def train_per_epoch(device, height, width, focal_len, cam_pose, near, far, sample, high_N, batch_size, model):
def train(images, poses, camera_info, args):
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--nEpochs', type=int, default=1500)
    Parser.add_argument('--samples', type=int, default=32)
    Parser.add_argument('--MiniBatchSize', type=int, default=4096)
    Parser.add_argument('--near', type=int, default=2)
    Parser.add_argument('--far',type=int, default=6)
    Parser.add_argument('--encode',type=int, default=6)

    Args = Parser.parse_args()
    num_epoch = Args.nEpochs
    sample = Args.samples
    batch_size = Args.MiniBatchSize
    near = Args.near
    far = Args.far
    high_N = Args.dim_encode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    focal_len = camera_info['f']
    height, width = all_images.shape[1:3]

    model = NeRF()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 5e-3)

    torch.manual_seed(9458)
    random.seed(9458)
    Loss = []
    Epochs = []
    for i in range(num_epoch+1):
        img_idx = random.randint(0, images.shape[0]-1)
        img = images[img_idx].to(device)
        img=img.float()
        cam_pose = poses[img_idx].to(device)
        rgb,_,_ = train_per_epoch(device, height, width, focal_len, cam_pose, near, far, sample, high_N, batch_size, model)
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
    SaveName =  './Checkpoint/' + 'model_' + str(i) + '.ckpt'
    torch.save({'epoch': Epochs, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, SaveName)
def test(images, poses, camera_info, args):
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
    if args.mode == 'train':
        print("Start training")
        train(images, poses, camera_info, args)
    elif args.mode == 'test':
        print("Start testing")
        args.load_checkpoint = True
        test(images, poses, camera_info, args)

def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default="./Phase2/Data/lego/",help="dataset path")
    parser.add_argument('--mode',default='train',help="train/test/val")
    parser.add_argument('--lrate',default=5e-4,help="training learning rate")
    parser.add_argument('--n_pos_freq',default=10,help="number of positional encoding frequencies for position")
    parser.add_argument('--n_dirc_freq',default=4,help="number of positional encoding frequencies for viewing direction")
    parser.add_argument('--n_rays_batch',default=32*32*4,help="number of rays per batch")
    parser.add_argument('--n_sample',default=400,help="number of sample per ray")
    parser.add_argument('--max_iters',default=10000,help="number of max iterations for training")
    parser.add_argument('--logs_path',default="./logs/",help="logs path")
    parser.add_argument('--checkpoint_path',default="./Phase2/example_checkpoint/",help="checkpoints path")
    parser.add_argument('--load_checkpoint',default=True,help="whether to load checkpoint or not")
    parser.add_argument('--images_path', default="./image/",help="folder to store images")
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)
