import cv2
import numpy as np
import argparse
import torch
import torch.nn.functional as F

from utils.utils import (
    image_to_tensor,
    disparity_to_tensor,
    render_3dphoto,
)
from model.AdaMPI import MPIPredictor

# DPT imports
from torchvision.transforms import Compose
from DPT.dpt.models import DPTDepthModel
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet
import DPT.util.io

def predict_depth_from_DPT(
    image_path, 
    height, 
    width, 
    DPT_optimize=False, 
    device='cuda',
):
    # DPT depth prediction
    DPT_model_path = 'DPT/weights/dpt_large-midas-2f21e586.pt'
    DPT_model = DPTDepthModel(
        path=DPT_model_path,
        backbone="vitl16_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    DPT_normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    net_w = width
    net_h = height
    DPT_transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=False,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                DPT_normalization,
                PrepareForNet(),
            ]
        )

    DPT_model.eval()
    if DPT_optimize == True and device == torch.device("cuda"):
        DPT_model = DPT_model.to(memory_format=torch.channels_last)
        DPT_model = DPT_model.half()
    DPT_model.to(device)

    DPT_img = DPT.util.io.read_image(image_path)
    # print('DPT_img shape: ', DPT_img.shape)
    DPT_img_input = DPT_transform({"image": DPT_img})["image"]
    # print('DPT_img_input shape: ', DPT_img_input.shape)

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(DPT_img_input).to(device).unsqueeze(0)

        if DPT_optimize == True and device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        prediction = DPT_model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=DPT_img_input.shape[1:3],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    depth = prediction
    depth_min = depth.min()
    depth_max = depth.max()

    if depth_max - depth_min > np.finfo("float").eps:
        out = (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.dtype)

    # print('out shape: ', out.shape)
    # print('out min: ', out.min())
    # print('out max: ', out.max())

    return out


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img_path', type=str, default="images/0810.png")
# parser.add_argument('--disp_path', type=str, default="images/depth/0810.png")
parser.add_argument('--width', type=int, default=384)
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--save_path', type=str, default="debug/0810.mp4")
parser.add_argument('--ckpt_path', type=str, default="adampiweight/adampi_64p.pth")
opt, _ = parser.parse_known_args()

# select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device: %s" % device)

# load input
image = image_to_tensor(opt.img_path).cuda()  # [1,3,h,w]
image = F.interpolate(image, size=(opt.height, opt.width), mode='bilinear', align_corners=True)
# print('image shape: ', image.shape)
# print('image device: ', image.device)
# disp = disparity_to_tensor(opt.disp_path).cuda()  # [1,1,h,w]
# disp = F.interpolate(disp, size=(opt.height, opt.width), mode='bilinear', align_corners=True)

# predict disparity from DPT model
disp = predict_depth_from_DPT(
    image_path=opt.img_path,
    height=opt.height,
    width=opt.width,
)
# save depth 
DPT.util.io.write_depth('debug/0810_depth', disp)
# cv2.imwrite('debug/0810_depth.png', disp)
# print('disp shape 1: ', disp.shape)
disp = torch.from_numpy(disp)[None, None, ...].float()
# print('disp shape 2: ', disp.shape)
disp = F.interpolate(disp, size=(opt.height, opt.width), mode='bilinear', align_corners=True)
# print('disp shape 3: ', disp.shape)
disp = disp.to(device)
# print('disp device: ', disp.device)


# load pretrained MPI model
ckpt = torch.load(opt.ckpt_path)
MPI_model = MPIPredictor(
    width=opt.width,
    height=opt.height,
    num_planes=ckpt["num_planes"],
)
MPI_model.load_state_dict(ckpt["weight"])
MPI_model = MPI_model.cuda()
MPI_model = MPI_model.eval()

# predict MPI planes
with torch.no_grad():
    pred_mpi_planes, pred_mpi_disp = MPI_model(image, disp)  # [b,s,4,h,w]

# render 3D photo
K = torch.tensor([
    [0.58, 0, 0.5],
    [0, 0.58, 0.5],
    [0, 0, 1]
]).cuda()
K[0, :] *= opt.width
K[1, :] *= opt.height
K = K.unsqueeze(0)

render_3dphoto(
    image,
    pred_mpi_planes,
    pred_mpi_disp,
    K,
    K,
    opt.save_path,
)
