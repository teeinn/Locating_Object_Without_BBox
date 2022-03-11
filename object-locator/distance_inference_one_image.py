from __future__ import print_function

__copyright__ = \
"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 10/02/2019 
"""
__license__ = "CC BY-NC-SA 4.0"
__authors__ = "Javier Ribera, David Guera, Yuhao Chen, Edward J. Delp"
__version__ = "1.6.0"


import argparse
import os
import sys
import time
import shutil
from parse import parse
import math
from collections import OrderedDict
import itertools

import matplotlib
matplotlib.use('Agg')
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import skimage.io
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import torchvision as tv
from torchvision.models import inception_v3
import skimage.transform
from peterpy import peter
from ballpark import ballpark

from data import csv_collator
from data import ScaleImageAndLabel
from data import build_dataset_personal
import losses
import argparser
from models import unet_model
from metrics import Judge
from metrics import make_metric_plots
import utils
import cv2
from aux_functions import *
import re

# Parse command line arguments
os.environ["CUDA_VISIBLE_DEVICES"]="1"
args = argparser.parse_command_args('testing')
# Tensor type to use, select CUDA or not
torch.set_default_dtype(torch.float32)
device_cpu = torch.device('cpu')
device = torch.device('cuda') if args.cuda else device_cpu
# Set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)
#--------------------------------------------------------------------------------------------------------------------------
# if os.path.exists(args.pathOut):
#     shutil.rmtree(args.pathOut)
# os.mkdir(args.pathOut)
#
# #mp4 to jpg files in a directory
# def extractImages(pathIn, pathOut):
#     vidcap = cv2.VideoCapture(pathIn)
#     height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     fps = int(vidcap.get(cv2.CAP_PROP_FPS))
#     success, image = vidcap.read()
#     count = 0
#     while success:
#         success, image = vidcap.read()
#         print('Read a new frame: ', success)
#         if success:
#             cv2.imwrite(pathOut + "frame%d.jpg" % count, image)
#             print(count)
#             count += 1
#
#     return height, width, fps
# height, width, fps = extractImages(args.pathIn, args.pathOut)

height, width, fps = 480, 640, 1
onlyfiles = [f for f in os.listdir(args.pathOut) if os.path.isfile(os.path.join(args.pathOut, f))]
onlyfiles.sort(key=lambda f: int(re.sub('\D', '', f)))
#--------------------------------------------------------------------------------------------------------------------------
mouse_pts = []
def get_mouse_points(event, x, y, flags, param):
    # Used to mark 4 points on the frame zero of the video that will be warped
    # Used to mark 2 points on the frame zero of the video that are 6 feet away
    global mouseX, mouseY, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        cv2.circle(image, (x, y), 10, (0, 255, 255), 10)
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))
        print("Point detected")
        print(mouse_pts)


def get_camera_perspective(img, src_points):
    IMAGE_H = img.shape[0]
    IMAGE_W = img.shape[1]
    src = np.float32(np.array(src_points))
    dst = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    return M, M_inv
#--------------------------------------------------------------------------------------------------------------------------
# Get video handle
origsize = (height, width)
scale_w = 1.2 / 2
scale_h = 4 / 2

SOLID_BACK_COLOR = (41, 41, 41)
# Setuo video writer
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_movie = cv2.VideoWriter("/home/qisens/2020.3~/locating_object_without_bbox/locating-objects-without-bboxes/Pedestrian_detect.avi", fourcc, fps, (width, height))
bird_movie = cv2.VideoWriter(
    "/home/qisens/2020.3~/locating_object_without_bbox/locating-objects-without-bboxes/Pedestrian_bird.avi", fourcc, fps, (int(width * scale_w), int(height * scale_h))
)
# Initialize necessary variables
frame_num = 0
total_pedestrians_detected = 0

cv2.namedWindow("image")
cv2.setMouseCallback("image", get_mouse_points)
num_mouse_points = 0
first_frame_display = True
#--------------------------------------------------------------------------------------------------------------------------
resized_size = np.array([args.height, args.width])
# Loss function
criterion_training = losses.WeightedHausdorffDistance(resized_height=args.height,
                                                      resized_width=args.width,
                                                      return_2_terms=True,
                                                      device=device)

# Restore saved checkpoint (model weights)
with peter("Loading checkpoint"):

    if os.path.isfile(args.model):
        if args.cuda:
            checkpoint = torch.load(args.model)
        else:
            checkpoint = torch.load(
                args.model, map_location=lambda storage, loc: storage)

        # Model
        if args.n_points is None:
            if 'n_points' not in checkpoint:
                # Model will also estimate # of points
                model = unet_model.UNet(3, 1,
                                        known_n_points=None,
                                        height=args.height,
                                        width=args.width,
                                        ultrasmall=args.ultrasmallnet)

            else:
                # The checkpoint tells us the # of points to estimate
                model = unet_model.UNet(3, 1,
                                        known_n_points=checkpoint['n_points'],
                                        height=args.height,
                                        width=args.width,
                                        ultrasmall=args.ultrasmallnet)
        else:
            # The user tells us the # of points to estimate
            model = unet_model.UNet(3, 1,
                                    known_n_points=args.n_points,
                                    height=args.height,
                                    width=args.width,
                                    ultrasmall=args.ultrasmallnet)


        # Parallelize
        if args.cuda:
            model = nn.DataParallel(model)
        model = model.to(device)

        # Load model in checkpoint
        if args.cuda:
            state_dict = checkpoint['model']
        else:
            # remove 'module.' of DataParallel
            state_dict = OrderedDict()
            for k, v in checkpoint['model'].items():
                name = k[7:]
                state_dict[name] = v

        model.load_state_dict(state_dict)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n\__ loaded checkpoint '{args.model}' "
              f"with {ballpark(num_params)} trainable parameters")
        # print(model)

    else:
        print(f"\n\__  E: no checkpoint found at '{args.model}'")
        exit(-1)

    tic = time.time()

start_process=time.time()
# Set the module in evaluation mode
model.eval()
# Accumulative histogram of estimated maps
bmm_tracker = utils.AccBetaMixtureModel()
# Empty output CSV (one per threshold)
df_outs = [pd.DataFrame() for _ in args.taus]
# --force will overwrite output directory
if args.force:
    shutil.rmtree(args.out)
end_process=time.time()

first_frame = True
dataset_loading_time=0
total_cal_time = 0
add_text_and_save_frame_time = 0
feed_forward_time = 0
produce_heat_map_time = 0
variable_and_paint_on_original_image_time = 0
get_perspective_and_bird_eye_view_time = 0
draw_polygon_time = 0
detect_six_feet = 0


start_total=time.time()
# for path, dirs, files in os.walk(args.pathOut):
tic = time.time()
for file in onlyfiles:
    image_path = os.path.join(args.pathOut, file)
    print(file)
    start_data=time.time()
    try:
        testset = build_dataset_personal(image_path,
                                transforms=transforms.Compose([
                                    ScaleImageAndLabel(size=(args.height,
                                                             args.width)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5)),
                                ]),
                                ignore_gt=not args.evaluate,
                                max_dataset_size=args.max_testset_size)
    except ValueError as e:
        print(f'E: {e}')
        exit(-1)


    dictionaries=[]
    imgs=testset[0][0].unsqueeze(0)
    dictionaries.append(testset[0][1])
    end_data=time.time()
    dataset_loading_time += (end_data-start_data)

    start_=time.time()
    start = time.time()
    # Move to device
    imgs = imgs.to(device)

    # Feed forward
    with torch.no_grad():
        est_maps, est_count = model.forward(imgs)

    # Convert to original size
    est_map_np = est_maps[0, :, :].to(device_cpu).numpy()
    est_map_np_origsize = \
        skimage.transform.resize(est_map_np,
                                 output_shape=origsize,
                                 mode='constant')
    orig_img_np = imgs[0].to(device_cpu).squeeze().numpy()
    orig_img_np_origsize = ((skimage.transform.resize(orig_img_np.transpose((1, 2, 0)),
                                                   output_shape=origsize,
                                                   mode='constant') + 1) / 2.0 * 255.0).\
        astype(np.float32).transpose((2, 0, 1))

    # Overlay output on original image as a heatmap
    orig_img_w_heatmap_origsize = utils.overlay_heatmap(img=orig_img_np_origsize,
                                                        map=est_map_np_origsize).\
        astype(np.float32)

    # Tensor -> int
    est_count_int = int(round(est_count.item()))

    n = height * width
    blocksize = 1024
    nblocks = np.float32(int(n / blocksize))

    # The estimated map must be thresholded to obtain estimated points
    for t, tau in enumerate(args.taus):
        if tau != -2:
            mask, _ = utils.threshold(est_map_np_origsize, tau, nblocks)
        else:
            mask, _, mix = utils.threshold(est_map_np_origsize, tau, nblocks)
            bmm_tracker.feed(mix)

        centroids_wrt_orig = utils.cluster(mask, est_count_int,
                                           max_mask_pts=args.max_mask_pts)
        end=time.time()
        feed_forward_time += (end-start)

        if args.paint:

            start=time.time()
            # Paint a cross at the estimated centroids
            img_with_x_n_map = utils.paint_circles(img=orig_img_w_heatmap_origsize,
                                                   points=centroids_wrt_orig,
                                                   color='red',
                                                   crosshair=True)
            os.makedirs(os.path.join(args.out,
                                     'intermediate',
                                     'painted_on_estimated_map',
                                     f'tau={round(tau, 4)}'), exist_ok=True)
            cv2.imwrite(os.path.join(args.out,
                                     'intermediate',
                                     'painted_on_estimated_map',
                                     f'tau={round(tau, 4)}',
                                     dictionaries[0]['filename']),
                        img_with_x_n_map.transpose((1, 2, 0))[:, :, ::-1])
            end=time.time()
            produce_heat_map_time += (end - start)

            start=time.time()
            img_with_x_n_map = utils.paint_circles(img=orig_img_np_origsize,
                                                   points=centroids_wrt_orig,
                                                   color='red',
                                                   crosshair=True)
            orig_img_np_origsize_dis= img_with_x_n_map.transpose((1, 2, 0))[:, :, ::-1]


            abs_six_feet_violations = 0
            frame_num += 1
            frame_h = orig_img_np_origsize_dis.shape[0]
            frame_w = orig_img_np_origsize_dis.shape[1]

            image = orig_img_np_origsize_dis.astype(np.uint8)
            pedestrian_detect = orig_img_np_origsize_dis.astype(np.uint8)
            end=time.time()
            variable_and_paint_on_original_image_time += (end-start)


            if first_frame:
                start__=time.time()
                first_frame = False
                while True:
                    cv2.imshow("image", image)
                    cv2.waitKey(1)
                    if len(mouse_pts) == 7:
                        cv2.destroyWindow("image")
                        break
                    first_frame_display = False
                four_points = mouse_pts


                start=time.time()
                # Get perspective
                M, Minv = get_camera_perspective(pedestrian_detect, four_points[0:4])
                pts = src = np.float32(np.array([four_points[4:]]))
                warped_pt = cv2.perspectiveTransform(pts, M)[0]
                d_thresh = np.sqrt(
                    (warped_pt[0][0] - warped_pt[1][0]) ** 2
                    + (warped_pt[0][1] - warped_pt[1][1]) ** 2
                )
                bird_image = np.zeros(
                    (int(frame_h * scale_h), int(frame_w * scale_w), 3), np.uint8
                )

                bird_image[:] = SOLID_BACK_COLOR
                end=time.time()
                get_perspective_and_bird_eye_view_time += (end-start)
                end__ = time.time()

            start=time.time()
            # draw polygon of ROI
            pts = np.array(
                [four_points[0], four_points[1], four_points[3], four_points[2]], np.int32)
            cv2.polylines(pedestrian_detect, [pts], True, (0, 255, 255), thickness=4)
            end=time.time()
            draw_polygon_time += (end - start)

            start=time.time()
            # Detect person and bounding boxes using DNN
            num_pedestrians = len(centroids_wrt_orig)
            if num_pedestrians > 0:
                # pedestrian_detect = plot_pedestrian_boxes_on_image(image, centroids_wrt_orig)
                warped_pts, bird_image = plot_points_on_bird_eye_view(
                    pedestrian_detect, centroids_wrt_orig, M, scale_w, scale_h
                )
                six_feet_violations, pairs = plot_lines_between_nodes(
                    warped_pts, bird_image, d_thresh
                )
                abs_six_feet_violations += six_feet_violations
            end=time.time()
            detect_six_feet += (end - start)

            start=time.time()
            last_h = 75
            text = "# 2m violations: " + str(int(abs_six_feet_violations))
            pedestrian_detect, last_h = put_text(pedestrian_detect, text, text_offset_y=last_h)

            text = "Count: {}".format(num_pedestrians)
            pedestrian_detect, last_h = put_text(pedestrian_detect, text, text_offset_y=last_h)

            cv2.imshow("Street Cam", pedestrian_detect)
            cv2.waitKey(3)
            try:
                print(pedestrian_detect.shape)
                output_movie.write(pedestrian_detect)
                bird_movie.write(bird_image)
            except:
                print("it's not working")
            end=time.time()
            add_text_and_save_frame_time += (end-start)
end_=time.time()
total_cal_time += (end_-start_)

elapsed_time = int(time.time() - tic)
end_total=time.time()

print("first process:{}".format(end_process-start_process))
print("mouse clicking:{}".format(end__-start__))
print("all the calculation including dataset loading:{}".format(end_total-start_total))
print("total_cal_time:{}".format(total_cal_time))
print("dataset_loading_time:{}".format(dataset_loading_time))

print("add_text_and_save_frame_time : {}".format(add_text_and_save_frame_time))
print("feed_forward_time : {}".format(feed_forward_time))
print("produce_heat_map_time : {}".format(produce_heat_map_time))
print("variable_and_paint_on_original_image_time : {}".format(variable_and_paint_on_original_image_time))
print("get_perspective_and_bird_eye_view_time : {}".format(get_perspective_and_bird_eye_view_time))
print("draw_polygon_time : {}".format(draw_polygon_time))
print("detect_six_feet : {}".format(detect_six_feet))

print(f'It took {elapsed_time} seconds to evaluate all this dataset.')