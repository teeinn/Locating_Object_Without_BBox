run visdom => python -m visdom.server
check visdom localhost number => for me, it was 8097
run training command => python -m object-locator.train --train-dir ./mall_dataset/train/ --batch-size 32 --lr 1e-4 --val-dir ./mall_dataset/val/ --optim Adam --save ./saved_model.ckpt --val-freq 1 --visdom-server http://localhost --visdom-port 8097 --visdom-env main
run evaluation command => python -m object-locator.locate --dataset ./mall_dataset/frames --out ./output --model ./saved_model_bk.ckpt

python -m object-locator.locate --dataset ./mall_dataset/val --out ./output --model ./saved_model_taulist_radius4_malldataset_lre-4.ckpt


combine two viedeos 
ffmpeg -i Pedestrian_bird.avi -i Pedestrian_bird.avi -filter_complex hstack output.avi
(more than 2 videos -> hstack=inputs=3)





--pathIn "/home/qisens/2020.3~/locating_object_without_bbox/locating-objects-without-bboxes/socialdistancing/video2.mp4" --pathOut "/home/qisens/2020.3~/locating_object_without_bbox/locating-objects-without-bboxes/video_frames/" --out /home/qisens/2020.3~/locating_object_without_bbox/locating-objects-without-bboxes/output --model /home/qisens/2020.3~/locating_object_without_bbox/locating-objects-without-bboxes/socialdistancing/saved_model_socialdistance_0707.ckpt --taus -1 --max-mask-pts 1000



--pathIn "/home/qisens/2020.3~/locating_object_without_bbox/locating-objects-without-bboxes/output.mp4" --pathOut "/home/qisens/2020.3~/locating_object_without_bbox/locating-objects-without-bboxes/video_frames/" --out /home/qisens/2020.3~/locating_object_without_bbox/locating-objects-without-bboxes/output --model /home/qisens/2020.3~/locating_object_without_bbox/locating-objects-without-bboxes/saved_model_taulist_radius4_malldataset_lre-4.ckpt --taus -1 --max-mask-pts 1000
