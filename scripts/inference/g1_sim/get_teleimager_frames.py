from teleimager.image_client import ImageClient
import rerun as rr

img_client = ImageClient("vlu-isaacsim.qut.edu.au", request_bgr=True)


rr.init("testing")
rr.spawn()
# Cool stuff in rerun that we can log the jpeg image directly :D
rr.log(
    "head_frame",
    rr.EncodedImage(contents=img_client.get_head_frame().jpg, media_type="image/jpeg"),
)
