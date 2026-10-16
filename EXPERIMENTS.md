# Experiments

The main experiment configuration is stored in `configs/paper_main.yaml`.

## CAIR settings

The default method settings are:

```text
warm-start episodes          40
CAGrad c                     0.7
lambda learning rate         1e-4
cosine smoothing beta        0.9
multiplier initial value     1.0
multiplier learning rate     1.0
lambda completion threshold  0.05
CAGrad optimization steps    20
CAGrad optimizer LR          2.0
CAGrad momentum              0.5
```

The applied imitation weight is refreshed every 10 episodes for manipulation tasks and every 5 episodes for locomotion tasks.

## Training budgets

```text
FetchPickAndPlace   10,000,000 steps
FetchPush           10,000,000 steps
FetchSlide          10,000,000 steps
Adroit Hammer        3,000,000 steps
Adroit Pen           8,000,000 steps
Locomotion             250,000 steps per task
```

Task-specific SAC, DDPG, and PPO hyperparameters are listed directly in `configs/paper_main.yaml`.

## Data

Fetch demonstrations can be generated with the scripts in `scripts/`. Locomotion demonstrations use the Minari dataset identifiers stored in the configuration. Adroit benchmark assets must be obtained separately and placed under `data/adroit/`.
