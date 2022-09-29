import os
import math
from PIL import Image
import cv2
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from moviepy.editor import ImageSequenceClip

from utils.mpi import mpi_rendering
from utils.mpi.homography_sampler import HomographySample

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def batch_inverse_3x3(M, check_singular=False):
    
    a = M[:, 0:1, 0:1]
    b = M[:, 0:1, 1:2]
    c = M[:, 0:1, 2:3]
    d = M[:, 1:2, 0:1]
    e = M[:, 1:2, 1:2]
    h = M[:, 2:3, 1:2]
    f = M[:, 1:2, 2:3]
    g = M[:, 2:3, 0:1]
    i = M[:, 2:3, 2:3]

    A = (e * i - f * h)
    B = -(d * i - f * g)
    C = (d * h - e * g)
    D = -(b * i - c * h)
    E = (a * i - c * g)
    F = -(a * h - b * g)
    G = (b * f - c * e)
    H = -(a * f - c * d)
    I = (a * e - b * d)

    detM = a * A + b * B + c * C

    # Check is slow, default to False
    if check_singular:
        if torch.any(torch.isclose(detM, torch.zeros_like(detM))):
            raise Exception('There exists singular matrix in input!')

    invM = torch.zeros_like(M)
    invM[:, 0:1, 0:1] = A
    invM[:, 0:1, 1:2] = D
    invM[:, 0:1, 2:3] = G
    invM[:, 1:2, 0:1] = B
    invM[:, 1:2, 1:2] = E
    invM[:, 1:2, 2:3] = H
    invM[:, 2:3, 0:1] = C
    invM[:, 2:3, 1:2] = F
    invM[:, 2:3, 2:3] = I

    invM = invM / detM

    return invM


def image_to_tensor(img_path, unsqueeze=True):
    rgb = transforms.ToTensor()(Image.open(img_path))
    if unsqueeze:
        rgb = rgb.unsqueeze(0)
    return rgb


def disparity_to_tensor(disp_path, unsqueeze=True):
    disp = cv2.imread(disp_path, -1) / (2 ** 16 - 1)
    disp = torch.from_numpy(disp)[None, ...]
    if unsqueeze:
        disp = disp.unsqueeze(0)
    return disp.float()


def interpolate_transformation_matrix(T, num_steps=90):
    def make_T(rot, trans):
        T = np.eye(4)
        T[:3,:3] = rot
        T[:3,3] = trans
        return T

    tgt_rot = T[0,:3,:3].detach().cpu().numpy()
    tgt_trans = T[0,:3,3].detach().cpu().numpy()

    key_rots = R.from_matrix(np.stack([np.eye(3), tgt_rot]))
    key_times = [0.0, 1.0]
    slerp = Slerp(key_times, key_rots)

    times = list(np.linspace(0,1,num_steps))
    interp_rots = slerp(times)
    lst_interp_rots = list(interp_rots.as_matrix())
    lst_interp_trans = [tgt_trans * time for time in times]
    lst_interp_T = [torch.from_numpy(make_T(rot, trans)).float() for (rot, trans) in zip(lst_interp_rots, lst_interp_trans)]
    
    return lst_interp_T


def gen_swing_path(num_frames=90, r_x=0.14, r_y=0., r_z=0.10):
    "Return a list of matrix [4, 4]"
    t = torch.arange(num_frames) / (num_frames - 1)
    poses = torch.eye(4).repeat(num_frames, 1, 1)
    poses[:, 0, 3] = r_x * torch.sin(2. * math.pi * t)
    poses[:, 1, 3] = r_y * torch.cos(2. * math.pi * t)
    poses[:, 2, 3] = r_z * (torch.cos(2. * math.pi * t) - 1.)
    return poses.unbind()


def render_3dphoto(
    src_imgs,  # [b,3,h,w]
    mpi_all_src,  # [b,s,4,h,w]
    disparity_all_src,  # [b,s]
    k_src,  # [b,3,3]
    k_tgt,  # [b,3,3]
    save_path,
):
    h, w = mpi_all_src.shape[-2:]
    device = mpi_all_src.device
    homography_sampler = HomographySample(h, w, device)
    # k_src_inv = torch.inverse(k_src)
    k_src_inv = batch_inverse_3x3(k_src)

    # preprocess the predict MPI
    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        homography_sampler.meshgrid,
        disparity_all_src,
        k_src_inv,
    )
    mpi_all_rgb_src = mpi_all_src[:, :, 0:3, :, :]  # BxSx3xHxW
    mpi_all_sigma_src = mpi_all_src[:, :, 3:, :, :]  # BxSx1xHxW
    _, _, blend_weights, _ = mpi_rendering.render(
        mpi_all_rgb_src,
        mpi_all_sigma_src,
        xyz_src_BS3HW,
        use_alpha=False,
        is_bg_depth_inf=False,
    )
    mpi_all_rgb_src = blend_weights * src_imgs.unsqueeze(1) + (1 - blend_weights) * mpi_all_rgb_src

    # render novel views
    swing_path_list = gen_swing_path()
    # swing_path_list = gen_swing_path(
    #     num_frames=90,
    #     r_x=0.7,
    #     r_y=0.0,
    #     r_z=0.5
    # )
    frames = []
    # for cam_ext in tqdm(swing_path_list):
    for cam_ext in swing_path_list:
        frame = render_novel_view(
            mpi_all_rgb_src,
            mpi_all_sigma_src,
            disparity_all_src,
            cam_ext.cuda(),
            k_src_inv,
            k_tgt,
            homography_sampler,
        )
        frame_np = frame[0].permute(1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
        frame_np = np.clip(np.round(frame_np * 255), a_min=0, a_max=255).astype(np.uint8)
        frames.append(frame_np)
    rgb_clip = ImageSequenceClip(frames, fps=30)
    rgb_clip.write_videofile(save_path, verbose=False, codec='mpeg4', logger=None, bitrate='2000k')


def render_towards_target_photo(
    src_imgs,  # [b,3,h,w]
    mpi_all_src,  # [b,s,4,h,w]
    disparity_all_src,  # [b,s]
    # k_src,  # [b,3,3]
    # k_tgt,  # [b,3,3]
    cam_src,
    cam_tgt,
    save_path,
):
    k_src = cam_src.K[..., :3, :3]
    k_tgt = cam_tgt.K[..., :3, :3]
    T_tgt_src = cam_tgt.Tcw.T
    # T_tgt_src = cam_tgt.Twc.T

    h, w = mpi_all_src.shape[-2:]
    device = mpi_all_src.device
    homography_sampler = HomographySample(h, w, device)
    # k_src_inv = torch.inverse(k_src)
    k_src_inv = batch_inverse_3x3(k_src)

    # preprocess the predict MPI
    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        homography_sampler.meshgrid,
        disparity_all_src,
        k_src_inv,
    )
    mpi_all_rgb_src = mpi_all_src[:, :, 0:3, :, :]  # BxSx3xHxW
    mpi_all_sigma_src = mpi_all_src[:, :, 3:, :, :]  # BxSx1xHxW
    _, _, blend_weights, _ = mpi_rendering.render(
        mpi_all_rgb_src,
        mpi_all_sigma_src,
        xyz_src_BS3HW,
        use_alpha=False,
        is_bg_depth_inf=False,
    )
    mpi_all_rgb_src = blend_weights * src_imgs.unsqueeze(1) + (1 - blend_weights) * mpi_all_rgb_src

    # render novel views
    # swing_path_list = gen_swing_path()
    # swing_path_list = [T_tgt_src]
    swing_path_list = interpolate_transformation_matrix(T_tgt_src)
    frames = []
    # for cam_ext in tqdm(swing_path_list):
    for cam_ext in swing_path_list:
        frame = render_novel_view(
            mpi_all_rgb_src,
            mpi_all_sigma_src,
            disparity_all_src,
            cam_ext.cuda(),
            k_src_inv,
            k_tgt,
            homography_sampler,
        )
        frame_np = frame[0].permute(1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
        frame_np = np.clip(np.round(frame_np * 255), a_min=0, a_max=255).astype(np.uint8)
        frames.append(frame_np)
    rgb_clip = ImageSequenceClip(frames, fps=30)
    rgb_clip.write_videofile(save_path, verbose=False, codec='mpeg4', logger=None, bitrate='2000k')


def render_novel_view(
    mpi_all_rgb_src,
    mpi_all_sigma_src,
    disparity_all_src,
    G_tgt_src,
    K_src_inv,
    K_tgt,
    homography_sampler,
):
    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        homography_sampler.meshgrid,
        disparity_all_src,
        K_src_inv
    )

    xyz_tgt_BS3HW = mpi_rendering.get_tgt_xyz_from_plane_disparity(
        xyz_src_BS3HW,
        G_tgt_src
    )

    tgt_imgs_syn, _, _ = mpi_rendering.render_tgt_rgb_depth(
        homography_sampler,
        mpi_all_rgb_src,
        mpi_all_sigma_src,
        disparity_all_src,
        xyz_tgt_BS3HW,
        G_tgt_src,
        K_src_inv,
        K_tgt,
        use_alpha=False,
        is_bg_depth_inf=False,
    )

    return tgt_imgs_syn
