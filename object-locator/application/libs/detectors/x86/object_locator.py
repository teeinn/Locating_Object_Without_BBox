import os
import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn
import numpy as np
from .inference_lib import losses
from .inference_lib import utils
from .models import unet_model
from peterpy import peter
from collections import OrderedDict
from ballpark import ballpark
import pandas as pd
from ..utils.fps_calculator import convert_infr_time_to_fps
from .inference_lib.data import build_dataset_personal
from .inference_lib.data import ScaleImageAndLabel
import time
from torchvision import transforms

os.environ["CUDA_VISIBLE_DEVICES"]="1"

class Detector:
    """
    Perform object detection with the given model. The model is a quantized tflite
    file which if the detector can not find it at the path it will download it
    from neuralet repository automatically.

    :param config: Is a ConfigEngine instance which provides necessary parameters.
    """

    def __init__(self, config):
        
        self.config = config
        self.model_name = self.config.get_section_dict('Detector')['Name']
        self.fps = None
        self.image_size = [int(i) for i in self.config.get_section_dict('Detector')['ImageSize'].split(',')]
        model_path = self.config.get_section_dict('Detector')['ModelPath']
        self.cuda = self.config.get_section_dict('Detector')['Cuda']
        self.tau = self.config.get_section_dict('Detector')['tau']
        self.total_pedestrians_detected = 0
        self.num_mouse_points = 0
        self.right_after_concatenate = True

        # Tensor type to use, select CUDA or not
        torch.set_default_dtype(torch.float32)
        if self.cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        # Set seeds
        np.random.seed(0)
        torch.manual_seed(0)
        if self.cuda:
            torch.cuda.manual_seed_all(0)

        # Loss function
        criterion_training = losses.WeightedHausdorffDistance(resized_height=self.image_size[0],
                                                              resized_width=self.image_size[1],
                                                              return_2_terms=True,
                                                              device=device)

        # Restore saved checkpoint (model weights)
        with peter("Loading checkpoint"):
            if os.path.isfile(model_path):
                if self.cuda:
                    checkpoint = torch.load(model_path)
                else:
                    checkpoint = torch.load(
                        model_path, map_location=lambda storage, loc: storage)

                # Model
                if 'n_points' not in checkpoint:
                    # Model will also estimate # of points
                    model = unet_model.UNet(3, 1,
                                            known_n_points=None,
                                            height=self.image_size[0],
                                            width=self.image_size[1],
                                            ultrasmall=False)

                else:
                    # The checkpoint tells us the # of points to estimate
                    model = unet_model.UNet(3, 1,
                                            known_n_points=checkpoint['n_points'],
                                            height=self.image_size[0],
                                            width=self.image_size[1],
                                            ultrasmall=False)
            
                # Parallelize
                if self.cuda:
                    model = nn.DataParallel(model)
                model = model.to(device)

                # Load model in checkpoint
                if self.cuda:
                    state_dict = checkpoint['model']
                else:
                    # remove 'module.' of DataParallel
                    state_dict = OrderedDict()
                    for k, v in checkpoint['model'].items():
                        name = k[7:]
                        state_dict[name] = v

                model.load_state_dict(state_dict)
                num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"\n\__ loaded checkpoint '{self.model}' "
                      f"with {ballpark(num_params)} trainable parameters")
                # print(model)

            else:
                print(f"\n\__  E: no checkpoint found at '{self.model}'")
                exit(-1)

        # Set the module in evaluation mode
        model.eval()
        # Accumulative histogram of estimated maps
        bmm_tracker = utils.AccBetaMixtureModel()
        # Empty output CSV (one per threshold)
        df_outs = [pd.DataFrame() for _ in self.taus]

        self.detection_model = model

    def inference(self, image_path):
 
        #result: a dictionary contains of [{"id": 0, "bbox": [x1, y1, x2, y2], "score":s%}, {...}, {...}, ...]
        
        # t_begin = time.perf_counter()
        # output_dict = self.detection_model(input_tensor)
        # inference_time = time.perf_counter() - t_begin  # Seconds
        #
        # # Calculate Frames rate (fps)
        # self.fps = convert_infr_time_to_fps(inference_time)
        #
        # boxes = output_dict['detection_boxes']
        # labels = output_dict['detection_classes']
        # scores = output_dict['detection_scores']
        #
        # class_id = int(self.config.get_section_dict('Detector')['ClassID'])
        # score_threshold = float(self.config.get_section_dict('Detector')['MinScore'])
        # result = []
        # for i in range(boxes.shape[1]):  # number of boxes
        #     if labels[0, i] == class_id and scores[0, i] > score_threshold:
        #         result.append({"id": str(class_id) + '-' + str(i), "bbox": boxes[0, i, :], "score": scores[0, i]})

       try:
            testset = build_dataset_personal(image_path,
                                             transforms=transforms.Compose([
                                                 ScaleImageAndLabel(size=(self.image_size[0],
                                                                          self.image_size[1])),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5),
                                                                      (0.5, 0.5, 0.5)),
                                             ]),
                                             ignore_gt=True,
                                             max_dataset_size=np.inf)
        except ValueError as e:
            print(f'E: {e}')
            exit(-1)

        imgs = testset[0][0].unsqueeze(0)
        dictionaries = []
        dictionaries.append(testset[0][1])

        # Move to device
        imgs = imgs.to(device)
        t_begin = time.perf_counter()
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
                                                          mode='constant') + 1) / 2.0 * 255.0). \
            astype(np.float32).transpose((2, 0, 1))

        # Overlay output on original image as a heatmap
        orig_img_w_heatmap_origsize = utils.overlay_heatmap(img=orig_img_np_origsize,
                                                            map=est_map_np_origsize). \
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

            if args.paint:
                # Paint a cross at the estimated centroids
                img_with_x_n_map = utils.paint_circles(img=orig_img_w_heatmap_origsize,
                                                       points=centroids_wrt_orig,
                                                       color='red',
                                                       crosshair=True)
                img_with_heatmap = img_with_x_n_map.transpose((1, 2, 0))[:, :, ::-1].astype(np.uint8)
                img_with_x_n_map = utils.paint_circles(img=orig_img_np_origsize,
                                                       points=centroids_wrt_orig,
                                                       color='red',
                                                       crosshair=True)
                orig_img_np_origsize_dis = img_with_x_n_map.transpose((1, 2, 0))[:, :, ::-1]

                abs_six_feet_violations = 0
                frame_h = orig_img_np_origsize_dis.shape[0]
                frame_w = orig_img_np_origsize_dis.shape[1]
                image = orig_img_np_origsize_dis.astype(np.uint8)
                pedestrian_detect = orig_img_np_origsize_dis.astype(np.uint8)

                if frame_num == 0:
                    start_mouse = time.time()
                    while True:
                        cv2.imshow("image", image)
                        cv2.waitKey(1)
                        if len(mouse_pts) == 7:
                            cv2.destroyWindow("image")
                            break
                        first_frame_display = False
                    four_points = mouse_pts

                    # Get perspective
                    M, Minv = get_camera_perspective(pedestrian_detect, four_points[0:4])
                    pts = src = np.float32(np.array([four_points[4:]]))
                    warped_pt = cv2.perspectiveTransform(pts, M)[0]
                    d_thresh = np.sqrt(
                        (warped_pt[0][0] - warped_pt[1][0]) ** 2
                        + (warped_pt[0][1] - warped_pt[1][1]) ** 2
                    )
                    end_mouse = time.time()

                frame_num += 1
                # draw polygon of ROI
                pts = np.array(
                    [four_points[0], four_points[1], four_points[3], four_points[2]], np.int32)
                cv2.polylines(pedestrian_detect, [pts], True, (0, 255, 255), thickness=4)

                # Detect person and bounding boxes using DNN
                num_pedestrians = len(centroids_wrt_orig)
                if num_pedestrians > 0:
                    # pedestrian_detect = plot_pedestrian_boxes_on_image(image, centroids_wrt_orig)
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
                result = np.hstack((bird_image_result, img_with_heatmap, pedestrian_detect))
                cv2.imshow("Street Cam", result)

                key = cv2.waitKey(1)
                if key == 27:
                    break

                if right_after_concatenate:
                    result_movie = cv2.VideoWriter(
                        "/home/qisens/2020.3~/locating_object_without_bbox/locating-objects-without-bboxes/Pedestrian_detect.avi",
                        fourcc, fps, (result.shape[1], result.shape[0]))
                    right_after_concatenate = False
                try:
                    result_movie.write(result)
                    # output_movie.write(pedestrian_detect)
                    # bird_movie.write(bird_image)
                except:
                    print("it's not working")
