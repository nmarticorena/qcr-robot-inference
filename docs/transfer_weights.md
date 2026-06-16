# Transfer Weights

You will often train the policies in a remote computer (either your desktop or the HPC facilities) to facilitate the deployment in rllab we provide the following pixi task `pixi run copy_weights` to copy all the experiments you desire to test on the real-robot:

The available options are:
```
╭─ options ─────────────────────────────────────────────────────────────╮
│ -h, --help         show this help message and exit                    │
│ --remote STR       (default: hpc)                                     │
│ --remote-root STR  (default: repos/qcr-robot-inference/saved_weights) │
│ --local-root PATH  (default: saved_weights)                           │
│ --epochs [INT [INT ...]]                                              │
│                    (default: 350 750)                                 │
│ --include-last, --no-include-last                                     │
│                    (default: False)                                    │
╰───────────────────────────────────────────────────────────────────────╯
```

This will copy the selected EMA checkpoint epochs, formatted as four digits, plus the latest weight by default. For example, `--epochs 50 350 750` transfers `ema_net_epoch_0050.pth`, `ema_net_epoch_0350.pth`, and `ema_net_epoch_0750.pth`. It will also store the `stats.pkl` for your normalisation metrics and the `experiment_config.yaml` to load state/action keys and which cameras to utilise.


Here we recommend to add a host alias to your ssh config

```
 ~/.ssh/config
Host <NAME>
    HostName <HOSTNAME>
    User <USERNAME>
    IdentityFile ~/.ssh/id_ed25519 
    ServerAliveInterval 30
```

Alternatively you can just run:
```
pixi run copy_weights --remote <USERNAME>@<HOSTNAME>
```

As a reminder the experiments are log as `task_name/expriment_name` all under the `saved_weights` directory.
