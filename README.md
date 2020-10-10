# Learning Group Structure and Disentangled Representations of Dynamical Environments

This repository is the official supporting code for the paper "Learning Disentangled Representations and Group Structure of Dynamical Environments", which was accepted to NeurIPS 2020. You can find the ArXiv version of the paper <a href="https://arxiv.org/abs/2002.06991">here</a>.

The code is structured in notebook form, where each notebook is used to reproduce the experiments described in our paper and produce the plots shown in our paper. The notebooks are named after the figure that they are used in; for example, fig2_flatland generates plots used in figure 2. 

### Overview 

We consider representation learning of dynamical environments to be the task of learning: (i) to encode static observations (i.e. images) in a latent space and (ii) how actions evolve this latent representation such that it remains faithful to the ground-truth in the observation space. Our approach, as formalised in our paper, is to use spherical latent spaces (where observations are encoded as unit-norm n-dimensional vectors) and represent actions as unitary (i.e. norm-preserving) rotations.

Moreover, we aim to find not just a faithful representation of the environment, but a disentangled representation as defined by <a href="https://arxiv.org/abs/1812.02230">Higgins et al. (2018)</a>. At a high level, a disentangled representation is one where the generative factors of the environment (for example translations and rotations in a 3D world) can be independently identified and modified in the latent space encoding. Our work proposes a regularisation that, when applied to the learned representations of actions, encourages this disentanglement.

# Requirements

Our code requires Python 3 and Jupyter Notebook. In addition to standard packages found in most python distributions (numpy, matplotlib, etc.), this code requires pytorch (for neural network training) and Pygame (for Flatland). To install pygame, first install its dependencies with:

`sudo apt install mercurial libfreetype6-dev libsdl-dev libsdl-image1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libportmidi-dev libavformat-dev libsdl-mixer1.2-dev libswscale-dev libjpeg-dev`

then install Pygame itself:

`pip3 install hg+http://bitbucket.org/pygame/pygame`

and finally install Pymunk:

`pip install pymunk`

# Running experiments

All experiments in our paper can be reproduced by running the notebooks in this repository. Please note that fig3_flatland and fig2_flatland are identical, since the same notebook produces results used in both figures.

### Where should I start?

We recommend starting with the fig2_flatland notebook, which includes a detailed walkthrough of what the code does in relation with our paper.

### Data and pre-trained models

We provide all raw test results and pre-trained models used for figure 6, which are located in src/flatland/flat_game/results. Figure 6 can be obtained by running the fig6_createplot notebook, which loads all the test results and plots them with the appropriate confidence interval.

We do not provide pre-trained models used for the other figures, since the notebooks are mostly self-contained and train and test models in a relatively short time. In particular, our results shown in figures 2 and 3 for Flatland and the gridworld can be reproduced in roughly 10 minutes on a laptop. Results for figures 4 and 5 take about 30 minutes to an hour to reproduce.

Data for the experiment shown in figure 4 can be generated automatically by running the teapot notebook and setting CREATE_DATASET to True. The data will then be saved in the teapot folder.
