#
# TensorflowHumanPoseEstimator.py
#
# This is based on run.py of the followingtf-pose-estimator
# https://github.com/gsethi2409/tf-pose-estimation

import os
import sys
import shutil
#import argparse
#import logging
import sys
import time
import pprint
import glob
from tf_pose import common
import cv2
import numpy as np
import traceback
import json

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from ConfigParser import ConfigParser

class TensorflowHumanPoseEstimator:

  def __init__(self, inference_conf):
    print("=== TensorflowHumanPoseEstimator")
    config    = ConfigParser(inference_conf)
    INFERENCE = "inference"
    self.images_dir  = config.get(INFERENCE, "images_dir")
    self.outputs_dir = config.get(INFERENCE, "outputs_dir")
    if not os.path.exists(self.images_dir):
        raise Exception("Not found " + self.images_dir)
    if os.path.exists(self.outputs_dir):
        shutil.rmtree(self.outputs_dir)
    if not os.path.exists(self.outputs_dir):
        os.makedirs(self.outputs_dir)

    resize = config.get(INFERENCE, "resize")
    model  = config.get(INFERENCE, "model")
    resize_out_ratio  = config.get(INFERENCE, "resize_out_ratio")
    self.w, self.h = model_wh(resize)
    self.e = None
    if self.w == 0 or self.h == 0:
        self.estimator = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
    else:
        self.estimator = TfPoseEstimator(get_graph_path(model), target_size=(self.w, self.h))

    self.upsample_size = resize_out_ratio
    
  def predict(self):
    print("=== predict images_dir {}".format(self.images_dir))

    image_files  = glob.glob(self.images_dir + "/*.jpg")
    image_files += glob.glob(self.images_dir + "/*.png")
    print("-- image_files {}".format(image_files))
  
    if len(image_files) == 0:
        raise Exception("Not found images files in images_dir {}".format(images_dir))
          
    for image_file in image_files:
      image = common.read_imgfile(image_file, None, None)
      if image is None:
        continue 
      
      t = time.time()
      humans = self.estimator.inference(image, 
                   resize_to_default=(self.w > 0 and self.h > 0), 
                   upsample_size=self.upsample_size)
      elapsed = time.time() - t

      print("inference image: {} in {} seconds.".format(image_file, elapsed))

      image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
     
      basename = os.path.basename(image_file)
      output_image = os.path.join(self.outputs_dir, basename)
      cv2.imwrite(output_image, image)
  

  
if __name__ == '__main__':
    
  inference_conf = "./inference.conf"
  try: 
    if len(sys.argv) == 2:
      inference_conf = sys.argv[1]
    if not os.path.exists(inference_conf):
      raise Exception("Not found " + inference_conf)
    classifier = TensorflowHumanPoseEstimator(inference_conf) # args)
    classifier.predict()

  except:
    traceback.print_exc()
