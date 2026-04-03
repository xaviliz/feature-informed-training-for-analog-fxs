# Feature-Informed Training of Neural Models for Analog Audio Effects

This repository implements feature-informed neural modeling techniques for emulating classic analog hardware, specifically the LA2A Compressor and the OD300 Overdrive. The project focuses on training neural networks to capture the non-linear characteristics and dynamic response of these units, leveraging feature conditioning for improved accuracy and control.

## Overview
This project explores the application of feature-informed neural networks to the modeling of analog audio effects. Unlike traditional "black box" approaches, this method incorporates auxiliary features (such as side-chain levels for compression or input gain for distortion) directly into the training pipeline to better inform the model about the device's internal state.

## Key Features

* Neural Emulation of LA2A: Models the electro-optical attenuation behavior and variable recovery times of the Teletronix LA2A leveling amplifier.
* Neural Emulation of OD300: Captures the asymmetric clipping and tone shaping characteristics of the Boss OD300 Overdrive/Distortion.
* Feature Integration: Implements techniques to feed control signals and side-chain information into the model architecture (e.g., via conditioning networks or auxiliary inputs) to enhance dynamic accuracy.

