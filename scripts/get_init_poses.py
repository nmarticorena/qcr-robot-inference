import os
import json
import numpy as np
import tyro


def get_home(dataset_path: str, /):
    episodes_dir = os.path.join(dataset_path, "episodes")

    episodes = sorted(
        [
            x
            for x in os.listdir(episodes_dir)
            if not x.startswith(".") and x.startswith("episode_")
        ],
        key=lambda x: int(x.split("_")[-1]),
    )

    homes = []

    for episode in episodes:
        data_path = os.path.join(episodes_dir, episode, "data.json")

        with open(data_path, "r") as f:
            data = json.load(f)

        left = data["data"][0]["states"]["left_arm"]["qpos"]
        right = data["data"][0]["states"]["right_arm"]["qpos"]

        home = np.array([*left, *right])
        homes.append(home)

    if len(homes) == 0:
        print("No episodes found.")
        return

    avg_home = np.mean(homes, axis=0)

    home = {"home": avg_home.tolist()}

    with open("home.json", "w") as f:
        json.dump(home, f, indent=2)
        

    print(avg_home)


if __name__ == "__main__":
    tyro.cli(get_home)
