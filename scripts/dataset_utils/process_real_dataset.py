import os
import h5py
import pims
import cv2
import numpy as np
import tqdm
import tyro

from rs_imle_policy.configs.train_config import VisionConfig


def convert_to_h5(dataset_path: str, /, vision_config: VisionConfig):
    dataset_name = dataset_path + "/images.h5"
    # Open an HDF5 file in write mode
    with h5py.File(dataset_name, "w") as h5f:
        episodes = sorted(os.listdir(os.path.join(dataset_path, "episodes")), key=int)
        for episode in tqdm.tqdm(episodes):
            grp = h5f.create_group(f"{episode}")
            for ix, cam_name in enumerate(vision_config.cameras):
                serial = vision_config.cameras_params[ix].serial_number
                video = os.path.join(dataset_path, "episodes", episode, "video", f"{serial}.mp4")

                pims_video = pims.PyAVReaderIndexed(video)
                frames = np.array([cv2.resize(np.array(frame), vision_config.img_shape[::-1]) for frame in pims_video])

                # Assuming typical access of 16 frames at a time.
                chunk_size = (16, *vision_config.img_shape, 3)

                # Store wrist and side video frames in separate datasets with chunking
                grp.create_dataset(cam_name, data=frames, compression="gzip", chunks=chunk_size)
    print(f"Video data saved to {dataset_name}")
    with open(dataset_path + "/vision_config.yml", "w") as f:
        f.write(tyro.extras.to_yaml(vision_config))


if __name__ == "__main__":
    tyro.cli(convert_to_h5)
