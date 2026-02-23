from teleimager.image_client import ImageClient
import rerun as rr
import cv2


class PerceptionSystem:
    """Manages camera perception for robot control.

    This class handles initialization and control of multiple RealSense cameras
    used for visual perception in robot inference tasks.

    Attributes:
        cams: MultiRealsense camera manager
        cams_config: Vision configuration parameters
    """

    def __init__(self, host, port):
        self.cams = ImageClient(host=host, request_port=port, request_bgr=True)

    def start(self):
        """Start the camera system and configure camera settings."""

    def stop(self):
        """Stop the camera system."""
        self.cams.stop()

    def get(self):
        """Get the latest frame"""
        bgr = self.cams.get_head_frame().bgr
        jpg = self.cams.get_head_frame().jpg
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rr.log("head_frame", rr.EncodedImage(contents=jpg, media_type="image/jpeg"))
        return rgb


class G1ArmsInferenceController:
    def __init__(self, host: str, port: str):
        self.perception_system = PerceptionSystem(host, port)
