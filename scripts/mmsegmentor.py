#!/usr/bin/env python3
# license removed for brevity

# !/home/leelingzhen/open_mmsegmentation/openmmlab/bin/python3

import rospy
import os
import numpy as np
from sensor_msgs.msg import Image


import mmseg
from mmseg.apis import inference_segmentor, init_segmentor
print(mmseg.__file__)

PALETTE = [
    [1, 1, 1],  # unknown
    [245, 254, 184],  # driveable surface
    [95, 235, 52],  # humans
    [52, 107, 235],  # moveable object
    [150, 68, 5],  # vehicles
]

CONFIG_DIR = os.path.realpath(os.path.dirname(__file__))
config_file = os.path.join(CONFIG_DIR, 'fcn_hr18_512x1024_160k_nuimages.py')
checkpoint_file = os.path.join(CONFIG_DIR, 'iter_160000.pth')

# build the model from a config file and a checkpoint file
MODEL = init_segmentor(config_file, checkpoint_file, device='cuda:0')
# # test a single image and show the results

BUFFER = 0


def image_callback(image_data):
    pub = rospy.Publisher('semantic_mask', Image, queue_size=10)
    # global BUFFER
    # BUFFER += 1
    #
    # if BUFFER == 5:

    # only way in converting image_callback to a numpyarray
    # taking image from image callback, image_data.data
    im = np.frombuffer(image_data.data, dtype=np.uint8).reshape(
        image_data.height, image_data.width, -1)
    im = im[:, :, 0:3]
    np_im = np.asarray(im)

    # getting the result from the model
    seg_mask = inference_segmentor(MODEL, np_im)

    seg_img = MODEL.show_result(
        img=im,
        result=seg_mask,
        palette=PALETTE,
        opacity=0.5,
    )

    # manually converting seg_img to the same data type as a image rostopic
    # best practice would be to use cv_bridge
    # https://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

    # assigning image_output to image_data
    # then manually changing the .data field
    image_output = image_data
    image_output.data = seg_img.tobytes()
    image_output.encoding = 'bgr8'

    # publish masks
    pub.publish(image_output)
    # BUFFER = 0

    return None


def main():

    rospy.init_node('semantic_masks', anonymous=True)
    rospy.Subscriber('/image_transport/image_decompressed',
                     Image, image_callback)

    rospy.spin()


if __name__ == '__main__':
    # try:
    #     talker()
    # except rospy.ROSInterruptException:
    main()
