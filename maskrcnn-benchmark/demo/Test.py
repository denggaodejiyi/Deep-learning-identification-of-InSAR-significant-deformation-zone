from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import cv2
import os
 
config_file = "../configs/e2e_mask_rcnn_R_50_FPN_1x.yaml"
 
# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
 
coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
 
file_root = '../datasets/coco/test2014/'
file_list = os.listdir(file_root)
save_out = "../output/"
for img_name in file_list:
    img_path = file_root + img_name
    image = cv2.imread(img_path)
    predictions = coco_demo.run_on_opencv_image(image)
    save_path = save_out + img_name
    cv2.imwrite(save_path,predictions)
 
    cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    cv2.imshow("img",predictions)
    cv2.waitKey(1)