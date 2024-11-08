# Snake-RL 🐍

This project implements a Reinforcement Learning (RL) agent to play the game of Snake using [Stable Baselines](https://stable-baselines3.readthedocs.io/). 

![Demo](media/Johnny-Dense-2.gif) <!-- Replace with your actual media path -->

## Table of Contents


- [Snake-RL 🐍](#snake-rl-)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Environment](#environment)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Training the agent](#training-the-agent)
  - [Evaluation](#evaluation)

---

## Overview

The Snake game is a classic reinforcement learning environment that poses challenges like exploration, decision-making, and handling delayed rewards. Using Stable Baselines, this project trains a simple RL agent to play the game decently well.

### Environment

This project slightly modifies the [Gym-Snake](https://github.com/grantsrb/Gym-Snake) repository for training. Thanks to [grantsrb](https://github.com/grantsrb) for making it. Apart from a couple of minor bugfixes / quality of life improvements, the main change was the addition of a small dense reward based on the manhattan distance between the head of the snake and the square containing the food. 

## Getting Started

### Prerequisites

- **Python** (>=3.7)
- **Stable Baselines 3**
- **Gymnasium**
- **Pytorch**
- **Numpy**
- **Matplotlib** (for plotting results)

### Installation

1. First install all dependencies with:

```bash
pip install -r requirements.txt
```
2. Then pip install the Gym-Snake environment locally

```bash
pip install -e Gym-Snake
```

## Training the agent

To train the RL agent, run:

```bash
python train_agent.py
```

Training outputs will be saved to the agents folder


## Evaluation
Once trained, the model can be evaluated with:

```bash
python eval_agent.py
```
This script will visualize the Snake agent’s performance.



