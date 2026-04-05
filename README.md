# Feature-Informed Training of Neural Models for Analog Audio Effects

This repository implements feature-informed neural modeling techniques for emulating classic analog hardware, specifically the LA2A Compressor and the OD300 Overdrive. The project focuses on training neural networks to capture the non-linear characteristics and dynamic response of these units, leveraging feature conditioning for improved accuracy and control.

## Table of Contents

<!-- TODO: fill this section at the end -->

- [Feature-Informed Training of Neural Models for Analog Audio Effects](#feature-informed-training-of-neural-models-for-analog-audio-effects)
  - [Table of Contents](#table-of-contents)
  - [General Info](#general-info)
  - [Key Features](#key-features)
  - [Folder Structure](#folder-structure)
  - [Installation](#installation)
  - [Datasets](#datasets)
    - [Teletronix LA2A Optical Compressor](#teletronix-la2a-optical-compressor)
    - [Behringer OD300 Overdrive](#behringer-od300-overdrive)
  - [Training](#training)

## General Info

This project explores the application of feature-informed neural networks to the modeling of analog audio effects. Unlike traditional "black box" approaches, this method incorporates auxiliary features (such as side-chain levels for compression or input gain for distortion) directly into the training pipeline to better inform the model about the device's internal state.

## Key Features

* Neural Emulation of LA2A: Models the electro-optical attenuation behavior and variable recovery times of the Teletronix LA2A leveling amplifier.
* Neural Emulation of OD300: Captures the asymmetric clipping and tone shaping characteristics of the Boss OD300 Overdrive/Distortion.
* Feature Integration: Implements techniques to feed control signals and side-chain information into the model architecture (e.g., via conditioning networks or auxiliary inputs) to enhance dynamic accuracy.

## Folder Structure

```bash
feature-informed-training-for-analog-fxs/
├── README.md
├── requirements.txt
├── scripts
│   ├── download_audio_effects.sh
│   └── download_optical_drc.sh
└── src
    ├── amplitude_dependent_gain.py
    ├── checkpoint_manager.py
    ├── config
    │   ├── la2a.json
    │   └── od300.json
    ├── dataloader.py
    ├── extract_features.py
    ├── losses.py
    ├── model
    │   ├── film.py
    │   ├── glu.py
    │   └── lstm.py
    ├── run.sh
    ├── starter.py
    ├── train.py
    └── utils
        ├── dir.py
        ├── extractor.py
        └── train.py
```

## Installation

1. Create a virtual environment

   ```bash
   python -m venv .env
   ```

2. Activate it

   ```bash
   source .env/bin/activate
   ```

3. Install requirements

   ```bash
   pip install -r requirements.txt
   ```

## Datasets

The following annotated datasets have been used:

1. [`Teletronix LA2A Optical Compressor`](https://www.kaggle.com/datasets/riccardosimionato/optical-dynamic-range-compressors-la-2a-cl-1b)
2. [`Behringer OD300 Overdrive`](https://www.kaggle.com/datasets/riccardosimionato/audio-effects-datasets-vol-1)

To download annotated datasets use our scripts:

### Teletronix LA2A Optical Compressor

```bash
bash scripts/download_optical_drc.sh
```

### Behringer OD300 Overdrive

```bash
bash scripts/download_audio_effects.sh
```

## Training

```bash
cd src
bash run.sh -n <gpu-id>
```

<!-- TODO: add separated training commands for LA2A and OD300 datasets -->

<!-- TODO: add inference commands for LA2A and OD300 -->