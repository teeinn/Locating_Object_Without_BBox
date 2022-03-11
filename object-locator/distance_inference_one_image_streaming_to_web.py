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



import os
import time
import shutil
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import transforms
import skimage.transform
from peterpy import peter
from ballpark import ballpark
from data import ScaleImageAndLabel
from data import build_dataset_personal
import argparser
from models import unet_model
import utils
import cv2
from aux_functions import *
import imutils

class distance(object):
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"]="1"
        args = argparser.parse_command_args('testing')
        # Tensor type to use, select CUDA or not
        torch.set_default_dtype(torch.float32)
        self.device_cpu = torch.device('cpu')
        self.device = torch.device('cuda') if args.cuda else self.device_cpu
        # Set seeds
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed_all(args.seed)

        if os.path.exists(args.pathOut):
            shutil.rmtree(args.pathOut)
        os.mkdir(args.pathOut)
        self.video_path = args.pathIn
        self.output_path = args.pathOut
        
        self.height=args.height
        self.width=args.width
        self.evaluate=args.evaluate
        self.max_testset_size = args.max_testset_size
        self.taus=args.taus
        self.max_mask_pts = args.max_mask_pts
        
        self.image = None
        self.mouse_pts=[]
        
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
                self.model = model.to(self.device)

                # Load model in checkpoint
                if args.cuda:
                    state_dict = checkpoint['model']
                else:
                    # remove 'module.' of DataParallel
                    state_dict = OrderedDict()
                    for k, v in checkpoint['model'].items():
                        name = k[7:]
                        state_dict[name] = v

                self.model.load_state_dict(state_dict)
                num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print(f"\n\__ loaded checkpoint '{args.model}' "
                      f"with {ballpark(num_params)} trainable parameters")
            else:
                print(f"\n\__  E: no checkpoint found at '{args.model}'")
                exit(-1)

        # Set the module in evaluation mode
        self.model.eval()
        # Accumulative histogram of estimated maps
        self.bmm_tracker = utils.AccBetaMixtureModel()
        # Empty output CSV (one per threshold)
        self.df_outs = [pd.DataFrame() for _ in args.taus]

    def get_mouse_points(self, event, x, y, flags, param):
        # Used to mark 4 points on the frame zero of the video that will be warped
        # Used to mark 2 points on the frame zero of the video that are 6 feet away
        global mouseX, mouseY, mouse_pts
        if event == cv2.EVENT_LBUTTONDOWN:
            mouseX, mouseY = x, y
            cv2.circle(self.image, (x, y), 10, (0, 255, 255), 10)
            # if "mouse_pts" not in globals():
            #     self.mouse_pts = []
            self.mouse_pts.append((x, y))
            print("Point detected")
            print(self.mouse_pts)
    
    
    def get_camera_perspective(self, img, src_points):
        IMAGE_H = img.shape[0]
        IMAGE_W = img.shape[1]
        src = np.float32(np.array(src_points))
        dst = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    
        M = cv2.getPerspectiveTransform(src, dst)
        M_inv = cv2.getPerspectiveTransform(dst, src)
    
        return M, M_inv
    def set_ui(self, ui):
        self.ui = ui
    def get_frame(self):
        
        # mp4 to jpg files in a directory
        vidcap = cv2.VideoCapture(self.video_path)
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Get video handle
        origsize = (height, width)
        scale_w = 1.2 / 2
        scale_h = 4 / 2

        # Initialize necessary variables
        # frame_num = 0
        first_frame=True

        if vidcap.isOpened():
            self.running_video=True
            while(self.running_video):
                tic=time.time()
                ret, frame = vidcap.read()
                image_path = os.path.join(self.output_path, "frame%d.jpg" % 0)
                if not ret:
                    print("end of the video file...")
                    vidcap.open(self.video_path)
                    continue
                    
                else:
                    cv2.imwrite(image_path, frame)

                try:
                    testset = build_dataset_personal(image_path,
                                            transforms=transforms.Compose([
                                                ScaleImageAndLabel(size=(self.height,
                                                                         self.width)),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5),
                                                                     (0.5, 0.5, 0.5)),
                                            ]),
                                            ignore_gt=not self.evaluate,
                                            max_dataset_size=self.max_testset_size)
                except ValueError as e:
                    print(f'E: {e}')
                    exit(-1)

                imgs=testset[0][0].unsqueeze(0)
                dictionaries = []
                dictionaries.append(testset[0][1])

                # Move to device
                imgs = imgs.to(self.device)
                # Feed forward
                with torch.no_grad():
                    est_maps, est_count = self.model.forward(imgs)

                # start=time.time()
                # Convert to original size
                est_map_np = est_maps[0, :, :].to(self.device_cpu).numpy()
                est_map_np_origsize = \
                    skimage.transform.resize(est_map_np,
                                             output_shape=origsize,
                                             mode='constant')


                # orig_img_np = imgs[0].to(self.device_cpu).squeeze().numpy()
                # orig_img_np_origsize = ((skimage.transform.resize(orig_img_np.transpose((1, 2, 0)),
                #                                                output_shape=origsize,
                #                                                mode='constant') + 1) / 2.0 * 255.0).\
                #     astype(np.float32).transpose((2, 0, 1))

                # Overlay output on original image as a heatmap
                # orig_img_w_heatmap_origsize = utils.overlay_heatmap(img=orig_img_np_origsize,
                #                                                     map=est_map_np_origsize).\
                    # astype(np.float32)
                # end=time.time()
                # print("map:{}".format(end-start))
                # Tensor -> int

                est_count_int = int(round(est_count.item()))
                n = height * width
                blocksize = 1024
                nblocks = np.float32(int(n / blocksize))

                start=time.time()

                # The estimated map must be thresholded to obtain estimated points
                for t, tau in enumerate(self.taus):
                    if tau != -2:
                        mask, _ = utils.threshold(est_map_np_origsize, tau, nblocks)
                    else:
                        mask, _, mix = utils.threshold(est_map_np_origsize, tau, nblocks)
                        self.bmm_tracker.feed(mix)
                    start_=time.time()
                    centroids_wrt_orig = utils.cluster(mask, est_count_int,
                                                       max_mask_pts=self.max_mask_pts)
                    end_=time.time()
                    print("centroids:{}".format(end_-start_))
                    end=time.time()
                    print("tau:{}".format(end-start))

                    # start=time.time()
                    # Paint a cross at the estimated centroids
                    # img_with_x_n_map = utils.paint_circles(img=orig_img_w_heatmap_origsize,
                    #                                        points=centroids_wrt_orig,
                    #                                        color='red',
                    #                                        crosshair=True)
                    # img_with_heatmap = img_with_x_n_map.transpose((1, 2, 0))[:, :, ::-1].astype(np.uint8)
                    # img_with_x_n_map = utils.paint_circles(img=orig_img_np_origsize,
                    #                                        points=centroids_wrt_orig,
                    #                                        color='red',
                    #                                        crosshair=True)
                    # orig_img_np_origsize_dis= img_with_x_n_map.transpose((1, 2, 0))[:, :, ::-1]
                    # print("heatmap{}".format(time.time()-start))
                    #
                    # # Save thresholded map to disk
                    # os.makedirs(os.path.join(self.output_path,
                    #                          'intermediate',
                    #                          'estimated_map_thresholded',
                    #                          f'tau={round(tau, 4)}'),
                    #             exist_ok=True)
                    # cv2.imwrite(os.path.join(self.output_path,
                    #                          'intermediate',
                    #                          'estimated_map_thresholded',
                    #                          f'tau={round(tau, 4)}',
                    #                          dictionaries[0]['filename']),
                    #             mask)


                    utils.paint_circles(img=frame,
                                       points=centroids_wrt_orig,
                                       color='iceblue',
                                       crosshair=True)
                    abs_six_feet_violations = 0
                    frame_h, frame_w = frame.shape[0], frame.shape[1]
                    self.image = frame.astype(np.uint8)
                    pedestrian_detect = frame.astype(np.uint8)

                    if first_frame:
                        # Setuo video writer
                        cv2.namedWindow("image")
                        cv2.setMouseCallback("image", self.get_mouse_points)
                        while True:
                            cv2.imshow("image", self.image)
                            cv2.waitKey(1)
                            if len(self.mouse_pts) == 7:
                                cv2.destroyWindow("image")
                                break
                            first_frame = False
                        four_points = self.mouse_pts

                        # Get perspective
                        M, Minv = get_camera_perspective(pedestrian_detect, four_points[0:4])
                        pts = src = np.float32(np.array([four_points[4:]]))
                        warped_pt = cv2.perspectiveTransform(pts, M)[0]
                        d_thresh = np.sqrt(
                            (warped_pt[0][0] - warped_pt[1][0]) ** 2
                            + (warped_pt[0][1] - warped_pt[1][1]) ** 2
                        )

                    start=time.time()
                    # frame_num += 1

                    # draw polygon of ROI
                    pts = np.array(
                        [four_points[0], four_points[1], four_points[3], four_points[2]], np.int32)
                    cv2.polylines(pedestrian_detect, [pts], True, (0, 255, 255), thickness=4)

                    # Detect person and bounding boxes using DNN
                    num_pedestrians = len(centroids_wrt_orig)
                    if num_pedestrians > 0:
                        warped_pts, bird_image = plot_points_on_bird_eye_view(
                            pedestrian_detect, centroids_wrt_orig, M, scale_w, scale_h
                        )
                        bird_image_result, six_feet_violations, pairs = plot_lines_between_nodes(
                            warped_pts, bird_image, d_thresh
                        )
                        abs_six_feet_violations += six_feet_violations

                    last_h = 75
                    text = "# 2m violations: " + str(int(abs_six_feet_violations))
                    pedestrian_detect, last_h = put_text(pedestrian_detect, text, text_offset_y=last_h)

                    text = "Count: {}".format(num_pedestrians)
                    pedestrian_detect, last_h = put_text(pedestrian_detect, text, text_offset_y=last_h)

                    bird_image_result = imutils.resize(bird_image_result, height=frame_h)
                    result = np.hstack((bird_image_result, pedestrian_detect))
                    # result = np.hstack((bird_image_result, img_with_heatmap, pedestrian_detect))
                    # result = imutils.resize(result, width=1200)
            
                    print("bird and result{}".format(time.time() - start))
                    # print("flag:{}, type:{} shape:{}".format(encoded_input_img, type(encoded_input_img), encoded_input_img.shape))
                    self.ui.update(result)
                    toc=time.time()
                    print("time: {}".format(toc-tic))
                    # print("update: {}".format(end-start))
            vidcap.release()
        self.running_video=False
        cv2.destroyAllWindows()
    

