# Getting Started

## Conda environment

### Create environment

```bash
conda create -n robowriter python=3.11
conda activate robowriter
```

### Install [MuJoCo](<https://github.com/google-deepmind/mujoco>)

```bash
pip install mujoco
```

### Download models

We have a modified UR5E robot arm model in this repo.

You can download other models in [Model Gallery](<https://mujoco.readthedocs.io/en/stable/models.html>) or just download the models you need.

If you want to use git:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

Or just download zip, which is recommended.
