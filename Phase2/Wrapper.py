import argparse
import glob
import random
import imageio
import torch
import matplotlib.pyplot as plt
import os
import json
import cv2
import numpy as np
from NeRFModel import *
import imageio.v3 as iio
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
torch.cuda.empty_cache()

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
        pose = data['frames']
        cam_pose = []
        for i in range(len(pose)):
            pose[i]['transform_matrix'] = torch.tensor(pose[i]['transform_matrix'])
            cam_pose.append(pose[i]['transform_matrix'])
            pose[i]['rotation'] = torch.tensor(pose[i]['rotation'])
            pose[i]['file_path'] = os.path.join(data_path, pose[i]['file_path'][2:])
            image_paths.append(pose[i]['file_path'] + ".png")
    
    images = []
    for i in range(len(image_paths)):
        img = iio.imread(image_paths[i])
        blended = img[..., :3]
        cv2.normalize(blended, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = cv2.resize(blended, (200,200), interpolation=cv2.INTER_AREA)
        images.append(torch.tensor(image))
    
    camera_info = { "H" : images[0].shape[0], "W" : images[0].shape[1],
                   "f" : torch.tensor(0.5*images[0].shape[1]/np.tan(0.5*camera_fov)).to(device)}
    return torch.stack(images), torch.stack(cam_pose), camera_info

def PixelToRay(camera_info, pose):
    """
    Input:
        camera_info: image width, height, camera matrix 
        pose: camera pose in world frame
    Outputs:
        ray origin and direction
    """
    width = camera_info['W']
    height = camera_info['H']
    focal_len = camera_info['f']
    grid_x, grid_y = torch.meshgrid(torch.arange(width).to(pose),torch.arange(height).to(pose))
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
    len = torch.linspace(near,far,sample).to(ray_ori)
    noise_shape = list(ray_ori.shape[:-1]) + [sample]
    len = len + torch.rand(noise_shape).to(ray_ori)*(far-near)/sample
    sample_pts = torch.tensor(ray_ori[..., None, :] + ray_dir[..., None, :]*len[..., :, None])
    return sample_pts, len

def render(depth, rays_ori, rgb, sigma):
    e_10 = torch.tensor([1e10], dtype = rays_ori.dtype, device = rays_ori.device)
    e_10 = e_10.expand(depth[...,:1].shape)
    delta_i= depth[...,1:] - depth[...,:-1]
    adjacent_dist = torch.concat((delta_i, e_10), dim = -1)
    alpha = 1.0 - torch.exp(-1 * sigma * adjacent_dist)
    wts = alpha * cumulative_product(1.0 - alpha + 1e-10)
    rgb_map = (wts[..., None] * rgb).sum(dim = -2)
    return wts, rgb_map

def cumulative_product(tensor) :
    product = torch.cumprod(tensor, dim=-1)
    product = torch.roll(product, 1, dims=-1)
    product[..., 0] = 1.0
    return product

def loss_fn(rgb, img):
    """
    Input:
        groundtruth: groundtruth rgb values
        prediction: predicted rgb values
    Outputs:
        loss
    """
    loss = torch.nn.functional.mse_loss(rgb,img)
    return loss

def plot_loss(num_epoch, loss,i,args):
    plt.figure(figsize=(10, 4))
    plt.plot(num_epoch, loss)
    plt.title("Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(args.data_path, "loss", f"Loss_{i}.png"))
    plt.close()

def train_per_epoch(camera_info, cam_pose, near, far, sample, high_N, batch_size, model):
    rays_dir, rays_ori = PixelToRay(camera_info, cam_pose)
    rays_pos, len = ray_sample(rays_ori, rays_dir, near, far, sample)
    flatten_ray_pos = torch.Tensor(rays_pos.reshape((-1,3)))
    out = model.pos_enc(flatten_ray_pos, high_N)
    out_batch = [out[i:i + batch_size] for i in range(0, out.shape[0], batch_size)]
    model_out = []
    for b in out_batch:
        model_out.append((model(b)))
    radiance = torch.cat(model_out, dim=0)
    unflatten = list(rays_pos.shape[:-1]) + [4]
    radiance_field = torch.reshape(radiance, unflatten)
    sigma = torch.relu(radiance_field[...,3])
    rgb = torch.sigmoid(radiance_field[...,:3])
    wts, rgb_map = render(len, rays_ori, rgb, sigma)
    return rgb_map

def train(images, poses, camera_info,args):
    num_epoch = args.max_iters
    sample = args.n_sample
    batch_size = args.n_rays_batch
    near = args.near
    far = args.far
    high_N = args.encode
    focal_len = camera_info['f']
    height, width = camera_info['H'], camera_info['W']

    # Create output directories
    os.makedirs(args.images_path, exist_ok=True)
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(os.path.join(args.data_path, "loss"), exist_ok=True)

    model = NeRFmodel()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lrate)

    random.seed()
    Loss = []
    Epochs = []
    for i in range(num_epoch+1):
        img_idx = random.randint(0, images.shape[0]-1)
        cam_pose = poses[img_idx].to(device)
        rgb = train_per_epoch(camera_info, cam_pose, near, far, sample, high_N, batch_size, model)
        img = images[img_idx].to(device)
        img=img.float()
        loss= loss_fn(rgb, img)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i%50==0:
            plt.imshow(rgb.detach().cpu().numpy())
            plt.title(f"Iteration {i}")
            plt.savefig(os.path.join(args.images_path, f"rgb_{i}.png"))
            Loss.append(loss.item())
            Epochs.append(i+1)
            plot_loss(Epochs, Loss,i,args)
            print(f'Iteration {i}, Loss: {loss.item():.6f}')
            checkpoint = os.path.join(args.checkpoint_path, f'model_{i}.ckpt')
            torch.save({'epoch': Epochs, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint)

def translation(t):
    matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ]
    return torch.tensor(matrix, dtype=torch.float32)

def rotation_phi(phi):
    matrix = [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ]
    return torch.tensor(matrix, dtype=torch.float32)

def rotation_theta(theta):
    matrix = [
        [np.cos(theta), 0, -np.sin(theta), 0],
        [0, 1, 0, 0],
        [np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1],
    ]
    return torch.tensor(matrix, dtype=torch.float32)

def spherical(theta, phi, t):
    c2w = translation(t)
    c2w = rotation_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rotation_theta(theta / 180.0 * np.pi) @ c2w
    c2w = torch.tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32) @ c2w
    return c2w

def psnr(imageA, imageB, max_pixel=255.0):
    def mse(imageA, imageB):
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        return err
    
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return float('inf')
    PSNR = 20 * np.log10(max_pixel / np.sqrt(mse_value))
    return PSNR

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
    def find_gt_pose_index(generated_theta, original_poses):
        differences = np.abs(original_poses - generated_theta)
        closest_index = np.argmin(differences)
        return closest_index
    
    # Load model
    model = NeRFmodel()
    
    # Try to load checkpoint from model_path
    checkpoint_path = os.path.join(args.model_path, 'coarse_model.pth')
    if os.path.exists(checkpoint_path):
        CheckPoint = torch.load(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")
        model.load_state_dict(CheckPoint['state_dict'])
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}")
        print("Using untrained model for testing")
    
    model.to(torch.float32)
    model = model.to(device)
    model.eval()
    
    # Test with first pose
    rgb_test = train_per_epoch(camera_info, poses[0], args.near, args.far, args.n_sample, args.encode, args.n_rays_batch, model)
    plt.imshow(rgb_test.detach().cpu().numpy())
    plt.title(f"Testing Image")
    plt.savefig("rgb_test.png")
    
    # Generate video frames
    frames = []
    psnr_values = []
    ssim_values = []
    
    for theta in (np.linspace(0.0, 360.0, 120)):
        c2w = spherical(theta,-30, 4.0)
        transform = c2w.to(device)
        rgb_values= train_per_epoch(camera_info, transform, args.near, args.far, args.n_sample, args.encode, args.n_rays_batch,model)
        
        # Convert to numpy for metrics
        rgb_np = rgb_values.detach().cpu().numpy()
        frames.append((255*np.clip(rgb_np,0,1)).astype(np.uint8))
        
        # Calculate metrics if we have ground truth images
        if len(images) > 0:
            # Use first image as reference for now
            image_gt = images[0].detach().cpu().numpy()
            image_gen = rgb_np
            
            psnr_value = psnr(image_gt, image_gen)
            ssim_value = ssim(image_gt, image_gen, data_range=image_gen.max() - image_gen.min(), multichannel=True, channel_axis=-1)
            
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
    
    # Save plots if we have metrics
    if psnr_values:
        plt.figure(figsize=(10, 4))
        plt.plot(range(len(psnr_values)), psnr_values, label='PSNR')
        plt.title('PSNR over frames')
        plt.savefig("./Results/PSNR_plot.png")
        plt.close()
        
        plt.figure(figsize=(10, 4))
        plt.plot(range(len(ssim_values)), ssim_values, label='SSIM')
        plt.title('SSIM over frames')
        plt.savefig("./Results/SSIM_plot.png")
        plt.close()
    
    # Save video
    imageio.mimwrite("lego_gif.mp4", frames, fps=30, quality=7, macro_block_size=None)
    print("Test completed. Video saved as lego_gif.mp4")

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
    parser.add_argument('--data_path',default="./data/lego",help="dataset path")
    parser.add_argument('--mode',default='train',help="train/test/val")
    parser.add_argument('--lrate',default=5e-4,help="training learning rate")
    parser.add_argument('--n_pos_freq',default=10,help="number of positional encoding frequencies for position")
    parser.add_argument('--n_dirc_freq',default=4,help="number of positional encoding frequencies for viewing direction")
    parser.add_argument('--n_rays_batch',default=32*32*4,help="number of rays per batch")
    parser.add_argument('--n_sample',default=16,help="number of sample per ray")
    parser.add_argument('--max_iters',default=10000,help="number of max iterations for training")
    parser.add_argument('--logs_path',default="./logs/",help="logs path")
    parser.add_argument('--checkpoint_path',default="./checkpoints/",help="checkpoints path")
    parser.add_argument('--load_checkpoint',default=True,help="whether to load checkpoint or not")
    parser.add_argument('--images_path', default="./images/",help="folder to store images")
    parser.add_argument('--near', type=int, default=2)
    parser.add_argument('--far',type=int, default=6)
    parser.add_argument('--encode',type=int, default=6)
    parser.add_argument('--model_path',default="./models/",help="path to save/load models")
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)