#!/usr/bin/env python3

import os
import hydra
from pdds.run_sde_train import run as run_trainable


@hydra.main(config_path="config", config_name="main")
def main(cfg):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # os.environ["WANDB_START_METHOD"] = "fork"
    os.environ["WANDB_START_METHOD"] = "thread"

    run_trainable(cfg)

    return


if __name__ == "__main__":
    main()
