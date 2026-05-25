import os
import time

import rerun as rr
from teleimager.image_client import ImageClient

# TODO: Add ip as environ in pixi
img_client = ImageClient(os.environ["G1_IP"], request_bgr=True)


rr.init("testing")
rr.spawn()


# Cool stuff in rerun that we can log the jpeg image directly :D
while True:
    rr.log(
        "head_frame",
        rr.EncodedImage(contents=img_client.get_head_frame().jpg, media_type="image/jpeg"),
    )
    time.sleep(1 / 30)  # 30Hz
