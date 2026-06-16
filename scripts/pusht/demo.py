# Test to check the wand video capabilities
import numpy as np
import wandb
from rs_imle_policy.envs.push_t import PushTImageEnv

env = PushTImageEnv()

# get first observation
obs, info = env.reset()
video = []

for i in range(200):
    action = (np.random.randn(2) + 1) * 255
    obs, reward, done, _, info = env.step(action)
    video.append(obs["image"])

video = (np.stack(video) * 255)
video = np.astype(video, np.uint8)

with wandb.init() as run:
    run.log({"video": wandb.Video(video, format="mp4", fps = 10)})




