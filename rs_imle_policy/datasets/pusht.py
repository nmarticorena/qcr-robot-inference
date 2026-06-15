
import time
import pickle
import numpy as np
from numpy.typing import NDArray
from pathlib import Path

from rs_imle_policy.datasets.base_dataset import BaseDataset
from rs_imle_policy.configs.train_config import PushTVisionConfig


class PushTDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        vision_config = PushTVisionConfig()
        super().__init__(load_images = False, 
                         vision_config = vision_config, 
                         skip_normalization_keys=("images",),
                         *args, 
                         **kwargs)

    def create_rlds_dataset(self) -> dict[int, dict[str, NDArray]]:
        with open(self.dataset_path / "pusht.pkl", 'rb') as f:
            rlds = pickle.load(f)
        # create a new entry to rlds for each episode to store position data which is a slice of the obs
        for episode in rlds.keys():
            pos_array = []
            for i in range(len(rlds[episode]['obs'])):
                pos_array.append(rlds[episode]['obs'][i][:2])
            
            rlds[episode]['state'] = pos_array
        rlds = self.add_padding(rlds, pad_before=self.obs_horizon-1, pad_after=self.pred_horizon)
        return rlds

    def add_padding(self, rlds, pad_before, pad_after):
        # for padding before the first element of the episode we pad with the first element
        # for padding after the last element of the episode we pad with the last element

        for episode in rlds.keys():
            # Extract episode data
            state = rlds[episode]['state']
            action = rlds[episode]['action']
            images = rlds[episode]['images']
            
            # Pre-padding
            pre_pad_pos = [state[0]] * pad_before
            pre_pad_action = [action[0]] * pad_before
            pre_pad_images = np.repeat(images[:1], pad_before, axis=0)

            
            # Post-padding
            post_pad_pos = [state[-1]] * pad_after
            post_pad_action = [action[-1]] * pad_after
            post_pad_images = np.repeat(images[-1:], pad_after, axis=0)
            
            # Apply padding
            rlds[episode]['state'] = np.array(pre_pad_pos + state + post_pad_pos)
            rlds[episode]['action'] = np.array(pre_pad_action + action.tolist() + post_pad_action)
            rlds[episode]['images'] = np.concatenate([pre_pad_images, images, post_pad_images], axis=0)
        
        return rlds


    


if __name__ == '__main__':
    dataset = PushTDataset(
                 dataset_path=Path('data/pusht'),
                 pred_horizon=16,
                 obs_horizon=2,
                 action_horizon=2,
                )
                 # dataset_percentage=1.0)
    
    idx=0
    while True:
        start_time = time.time()
        dataset.__getitem__(idx)
        breakpoint()
        print(idx)
        # pdb.set_trace()
        # print(f"Time taken: {time.time() - start_time}")
        idx += 1
