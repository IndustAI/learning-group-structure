# Learning Group Structure and Disentangled Representations of Dynamical Environments

This folder contains notebooks that can be used to reproduce the experiments in our paper "Learning Disentangled Representations and Group Structure of Dynamical Environments", which was accepted to NeurIPS 2020. You can find the ArXiv version of our paper <a href="https://arxiv.org/abs/2002.06991">here</a>.

# Requirements

Our code runs using Python 3. In addition to standard packages found in most python distributions (numpy, matplotlib, etc.), this code requires pytorch (for neural network training) and Pygame (for Flatland). To install pygame, first install its dependencies with:

`sudo apt install mercurial libfreetype6-dev libsdl-dev libsdl-image1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libportmidi-dev libavformat-dev libsdl-mixer1.2-dev libswscale-dev libjpeg-dev`

then install Pygame itself:

`pip3 install hg+http://bitbucket.org/pygame/pygame`

and finally install Pymunk:

`pip install pymunk`

# Running experiments

All experiments in our paper can be reproduced by running the notebooks in this folder. The notebooks are named after the figure that they are used in; for example, fig2_flatland generates plots used in figure 2. 

Note: fig3_flatland and fig2_flatland are identical, since the same notebook produces results used in both figures.

# Where should I start?

We recommend starting with the fig2_flatland notebook, which includes a detailed walkthrough of what the code does in relation with our theory.

# Data and pre-trained models

We provide all raw test results and pre-trained models used for figure 6, which are located in src/flatland/flat_game/results. Figure 6 can be obtained by running the fig6_createplot notebook, which loads all the test results and plots them with the appropriate confidence interval.

We do not provide pre-trained models used for the other figures, since the notebooks are mostly self-contained and train and test models in a relatively short time. In particular, our results shown in figures 2 and 3 for Flatland and the gridworld can be reproduced in roughly 10 minutes on a laptop. Results for figures 4 and 5 take about 30 minutes to an hour to reproduce.

Data for the experiment shown in figure 4 can be generated automatically by running the teapot notebook and setting CREATE_DATASET to True. The data will then be saved in the teapot folder.
