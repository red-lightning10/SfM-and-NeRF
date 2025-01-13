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
import imageio.v3 as iio
from skimage.metrics import structural_similarity as ssim
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
if torch.cuda.is_available():
    # Set the device to GPU
    torch.cuda.set_device(0)
#    device = torch.device("cpu")
torch.cuda.empty_cache()
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
        #img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])
        #img = img[:,:3A]
        #print(img)
        # img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])
        # Check if image has an alpha channel
            # Separate RGB nd alpha channels
            # Perform alpha blending with a white backgro
        blended = img[..., :3]
        cv2.normalize(blended, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #print(blended)#blended = blended/255.0# For images without alpha channel
        # blended = blended.astype(np.uint8)
        # images.append(torch.tensor(image))
        image = cv2.resize(blended, (200,200), interpolation=cv2.INTER_AREA)
        cv2.imwrite("image.jpg",image)
        images.append(torch.tensor(image))
    camera_info = { "H" : images[0].shape[0], "W" : images[0].shape[1],
                   "f" : torch.tensor(0.5*images[0].shape[1]/np.tan(0.5*camera_fov)).to(device)}
    focal_len = camera_info['f']
    return torch.stack(images), torch.stack(cam_pose),camera_info


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
    # print(depth_values.shape)
    # print('------------DEPTH VALUES BEFORE NOISE----------------------------')
    noise_shape = list(ray_ori.shape[:-1]) + [sample]
    # print(noise_shape)
    len = len + torch.rand(noise_shape).to(ray_ori)*(far-near)/sample
    #len = len.to(device)
    # print(depth_values.shape)
    sample_pts = torch.tensor(ray_ori[..., None, :] + ray_dir[..., None, :]*len[..., :, None])
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
    plt.savefig(args.data_path+"/loss/Loss"+str(i)+".png")

def train_per_epoch(camera_info, cam_pose, near, far, sample, high_N, batch_size, model):
    rays_dir, rays_ori = PixelToRay(camera_info, cam_pose)
    rays_pos, len = ray_sample(rays_ori, rays_dir, near, far, sample)
    flatten_ray_pos = torch.Tensor(rays_pos.reshape((-1,3)))
    #rays_pos = rays_pos.reshape(-1,3)
    #flatten_ray_pos = rays_pos.clone().detach()
    out = model.pos_enc(flatten_ray_pos, high_N)
    out_batch = [out[i:i + batch_size] for i in range(0, out.shape[0], batch_size)]
    #rays_dir.detach()
    #rays_ori.detach()
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
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    focal_len = camera_info['f']
    #focal_len = focal_len.to(device)
    height, width = camera_info['H'], camera_info['W']

    model = NeRFmodel()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 10e-3)

    random.seed()
    Loss = []
    Epochs = []
    for i in range(num_epoch+1):
        img_idx = random.randint(0, images.shape[0]-1)
        #img = images[img_idx].to(device)
        #img=img.float()
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
            plt.savefig(args.data_path+"/rgb"+str(i)+".png")
            Loss.append(loss.item())
            Epochs.append(i+1)
            plot_loss(Epochs, Loss,i,args)
            print('i',i)
            checkpoint =  args.data_path+'/Checkpoint/' + 'model_' + str(i) + '.ckpt'
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
def translation(t):
    matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ]
    return torch.tensor(matrix, dtype=torch.float32)

# Rotation matrix for movement in phi
def rotation_phi(phi):
    matrix = [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ]
    return torch.tensor(matrix, dtype=torch.float32)

# Rotation matrix for movement in theta
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
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])

        return err
    # Assume the images are numpy arrays with the same dimension
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        # MSE is zero means no difference between images; PSNR is infinite
        return float('inf')
    # Calculate PSNR
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
        # Compute the absolute difference between the generated theta and all original poses
        differences = np.abs(original_poses - generated_theta)
        # Find the index of the smallest difference
        closest_index = np.argmin(differences)
        return closest_index
    # model = model_NeRF()
    model = NeRFmodel()
    CheckPoint = torch.load('/home/badari/ProjectTest/Phase2/Phase2/tarun/data/lego1/coarse_model.pth')
    print(CheckPoint)
    model.load_state_dict(CheckPoint['state_dict'])
    model.to(torch.float32)
    model = model.to(device)
    model.eval()
    rgb_test = train_per_epoch(camera_info, poses[0], args.near, args.far, args.n_sample, args.encode, args.n_rays_batch, model)
    plt.imshow(rgb_test.detach().cpu().numpy())
    plt.title(f"Testing Image")
    plt.savefig("rgb_test.png")
    frames = []
    for theta in (np.linspace(0.0, 360.0, 120)):
        c2w = spherical(theta,-30, 4.0)
        transform = c2w.to(device)
        rgb_values= train_per_epoch(camera_info, transform, args.near, args.far, args.n_sample, args.encode, args.n_rays_batch,model)
        closest_pose_index = find_gt_pose_index(theta, poses)
            # Get the original image that matches the generated image's pose
        image_gt = images_og[closest_pose_index].astype(np.float32)
        image1 = image.astype(np.float32)  # Assuming 'image' is your generated image
        psnr_value = psnr(image_gt, image1)
        ssim_value = ssim(image_gt, image1, data_range=image1.max() - image1.min(), multichannel=True, channel_axis=-1)  # Assuming RGB images

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        frames.append((255*np.clip(rgb_values.detach().cpu().numpy(),0,1)).astype(np.uint8))
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(psnr_values)), psnr_values, label='PSNR')
    plt.savefig("./Results/PSNR_plot.png")
    plt.close()
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(ssim_values)), ssim_values, label='PSNR')
    plt.savefig("./Results/SSIM_plot.png")
    plt.close()
    imageio.mimwrite("lego_gif.mp4", frames, fps=30, quality=7, macro_block_size=None)


def main(args):
    
    # load data
    print("Loading data...")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, poses, camera_info = loadDataset(args.data_path, args.mode)
    #im = images[0]
    #cv2.imwrite("image.jpg",im)#images = torch.tensor(images).to(device)
    #images.clone().detach()
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
    parser.add_argument('--data_path',default="/home/badari/ProjectTest/Phase2/Phase2/lego",help="dataset path")
    parser.add_argument('--mode',default='train',help="train/test/val")
    parser.add_argument('--lrate',default=5e-4,help="training learning rate")
    parser.add_argument('--n_pos_freq',default=10,help="number of positional encoding frequencies for position")
    parser.add_argument('--n_dirc_freq',default=4,help="number of positional encoding frequencies for viewing direction")
    parser.add_argument('--n_rays_batch',default=32*32*4,help="number of rays per batch")
    parser.add_argument('--n_sample',default=16,help="number of sample per ray")
    parser.add_argument('--max_iters',default=10000,help="number of max iterations for training")
    parser.add_argument('--logs_path',default="/home/badari/ProjectTest/Phase2/Phase2/ship-20240310T050938Z-001/logs/",help="logs path")
    parser.add_argument('--checkpoint_path',default="/home/badari/ProjectTest/Phase2/Phase2/ship-20240310T050938Z-001/checkpoints/",help="checkpoints path")
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
