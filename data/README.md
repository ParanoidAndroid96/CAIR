# Data

Large benchmark datasets and expert-policy files are not tracked in Git.

Fetch demonstration trajectories can be generated with `scripts/generate_fetch_demos.py`.

For Adroit experiments, place the required files in `data/adroit/`:

```text
hammer-demos-v0-bc-combined.hdf5
expert.zip
pen-v0_demos.pickle
pen-v0.pickle
```

Locomotion datasets are downloaded through Minari using `scripts/download_minari.py`.
