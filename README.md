# Learning Stable Deep Dynamics Models

Companion code to "Learning Stable Deep Dynamics Models" (Manek and Kolter, 2019)

## Installation

You need Python 3.6 or later, with packages listed in `requirements.txt`. Ensure that the directories `experiments/` and `runs/` are writable.

## Running

All training commands produce output in the `experiments/` folder, automatically named with the hyperparameters used in training. You can track training progress using `tensorboard runs/`.

### Pendulum

To train the models for an `<n>`-link pendulum, you should first run the command:

```sh
./train_pendulum_simple <n>
```

which will create and cache the training data and the evaluation data. (Warning: this is a cpu-heavy task that may take hours to complete.) Once that is complete, you can train multiple models in parallel with:

```sh
./train_many <n>
```

You can concurrently train train models with different `<n>`.

Ensure that `pendulum-cache/` is writable.

### VAE

To train the VAE, you need to convert videos to sequentially-numbered frames in `youtube/<video>/fr_[0-9]*.jpg`. A good place to start is [this](https://www.youtube.com/watch?v=I4dYM1biVq0) video, which you can download and convert with:

```sh
mkdir youtube
cd youtube
youtube-dl -o bonfire.%(ext) -f 134 I4dYM1biVq0
ffmpeg -ss 0:01:00 -i bonfire.mp4 -t 3:20 -vf scale=320:240 bonfire/fr_%06d.png
```

(You need to install `youtube-dl` and `ffmpeg`.)

Once that is done, you can train the simple VAE model with:

```sh
./train_vae_simple bonfire
```

And the stable VAE with:

```sh
./train_vae_stable stable_exp bonfire 0.0005 -0.25 0.00 PSICNN 0.001
```
