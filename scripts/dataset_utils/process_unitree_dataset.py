import os
import h5py
import cv2
import tqdm
import tyro

from rs_imle_policy.configs.train_config import G1VisionConfig


def convert_to_h5(dataset_path: str, /, vision_config: G1VisionConfig):
    dataset_name = dataset_path + "/images.h5"
    # Open an HDF5 file in write mode
    with h5py.File(dataset_name, "w") as h5f:
        episodes = sorted(
            [
                x
                for x in os.listdir(os.path.join(dataset_path, "episodes"))
                if not x.startswith(".") and x.startswith("episode_")
            ],
            key=lambda x: int(x.split("_")[-1]),
        )
        for episode in tqdm.tqdm(episodes):
            episode_id = int(episode.split("_")[-1])
            grp = h5f.create_group(f"{episode_id}")
            for ix, cam_name in enumerate(vision_config.cameras):
                image_path = os.path.join(dataset_path, "episodes", episode, "colors")
                image_files = sorted(os.listdir(image_path), key=lambda x: int(x.split("_")[0]))
                frames = []
                for image_file in image_files:
                    image = cv2.imread(os.path.join(image_path, image_file))
                    resized_image = cv2.resize(image, vision_config.img_shape[::-1])
                    frames.append(resized_image)

                # Assuming typical access of 16 frames at a time.
                chunk_size = (16, *vision_config.img_shape, 3)

                # Store wrist and side video frames in separate datasets with chunking
                grp.create_dataset(cam_name, data=frames, compression="gzip", chunks=chunk_size)
    print(f"Image data saved to {dataset_name}")
    with open(dataset_path + "/vision_config.yml", "w") as f:
        f.write(tyro.extras.to_yaml(vision_config))


if __name__ == "__main__":
    tyro.cli(convert_to_h5)
