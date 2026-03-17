import os
import tqdm
import pims
import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def adjust_hsv_ranges(image_path: str):
    # Load the image and convert to HSV
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Initial HSV ranges
    hsv_ranges = {
        "H_low": 0,
        "H_high": 255,
        "S_low": 130,
        "S_high": 216,
        "V_low": 150,
        "V_high": 230,
    }

    # Create figure and axis
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.4)
    mask = cv2.inRange(
        hsv_img,
        (hsv_ranges["H_low"], hsv_ranges["S_low"], hsv_ranges["V_low"]),
        (hsv_ranges["H_high"], hsv_ranges["S_high"], hsv_ranges["V_high"]),
    )
    result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    masked_img = ax.imshow(result)

    # Define slider axes
    axcolor = "lightgoldenrodyellow"
    ax_h_low = plt.axes([0.25, 0.3, 0.65, 0.03], facecolor=axcolor)
    ax_h_high = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
    ax_s_low = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
    ax_s_high = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_v_low = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_v_high = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

    # Create sliders
    s_h_low = Slider(ax_h_low, "H Low", 0, 255, valinit=hsv_ranges["H_low"], valstep=1)
    s_h_high = Slider(ax_h_high, "H High", 0, 255, valinit=hsv_ranges["H_high"], valstep=1)
    s_s_low = Slider(ax_s_low, "S Low", 0, 255, valinit=hsv_ranges["S_low"], valstep=1)
    s_s_high = Slider(ax_s_high, "S High", 0, 255, valinit=hsv_ranges["S_high"], valstep=1)
    s_v_low = Slider(ax_v_low, "V Low", 0, 255, valinit=hsv_ranges["V_low"], valstep=1)
    s_v_high = Slider(ax_v_high, "V High", 0, 255, valinit=hsv_ranges["V_high"], valstep=1)

    # Update function for sliders
    def update(val):
        # Get current slider values
        h_low = s_h_low.val
        h_high = s_h_high.val
        s_low = s_s_low.val
        s_high = s_s_high.val
        v_low = s_v_low.val
        v_high = s_v_high.val

        # Create new mask and update the displayed image
        lower_bound = np.array([h_low, s_low, v_low])
        upper_bound = np.array([h_high, s_high, v_high])
        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
        result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
        masked_img.set_data(result)
        fig.canvas.draw_idle()

    # Connect update function to sliders
    s_h_low.on_changed(update)
    s_h_high.on_changed(update)
    s_s_low.on_changed(update)
    s_s_high.on_changed(update)
    s_v_low.on_changed(update)
    s_v_high.on_changed(update)

    # Show the plot
    plt.show()

    # Print final HSV values after closing the plot
    print(
        f"Final HSV Ranges: H:({s_h_low.val}, {s_h_high.val}), S:({s_s_low.val}, {s_s_high.val}), V:({s_v_low.val}, {s_v_high.val})"
    )


def cache_pims_video(dataset_path):
    # Open an HDF5 file in write mode
    with h5py.File("t_block_1.h5", "w") as h5f:
        episodes = sorted(os.listdir(os.path.join(dataset_path, "episodes")), key=lambda x: int(x))
        for episode in tqdm.tqdm(episodes):
            video_wrist = os.path.join(dataset_path, "episodes", episode, "video", "0.mp4")
            video_side = os.path.join(dataset_path, "episodes", episode, "video", "2.mp4")

            # Use PIMS to read the video files
            pims_video_wrist = pims.PyAVReaderIndexed(video_wrist)
            pims_video_side = pims.PyAVReaderIndexed(video_side)

            # Convert frames to NumPy arrays and resize
            wrist_frames = np.array([cv2.resize(np.array(frame), (320, 240)) for frame in pims_video_wrist])
            side_frames = np.array([cv2.resize(np.array(frame), (320, 240)) for frame in pims_video_side])

            # Create a group for each episode in HDF5
            grp = h5f.create_group(f"{episode}")

            # Chunking size is critical: Use (chunk_frames, height, width, channels)
            # Assuming typical access of 16 frames at a time.
            chunk_size = (16, 240, 320, 3)

            # Store wrist and side video frames in separate datasets with chunking
            grp.create_dataset("wrist", data=wrist_frames, compression="gzip", chunks=chunk_size)
            grp.create_dataset("side", data=side_frames, compression="gzip", chunks=chunk_size)

    print("Video data saved to video_data.h5")
