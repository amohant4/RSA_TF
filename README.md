# RRAM Simulator Neural Network Training
This is an tensorflow implementation of neural network training and testing for RRAM based hardware accelerators.
This is a simulator to emulate RRAM device variation, read/write noise and its effect on neural network inference.  

### Requirements: software

1. Requirements for Tensorflow (see: [Tensorflow](https://www.tensorflow.org/))

2. Python packages you might not have: `cython`, `easydict`

### Requirements: hardware

1. For training, 3G of GPU memory is sufficient (using CUDNN)

### Installation

1. Clone the rramtraining (make sure to clone with --recursive)
  ```
  git clone --recursive https://amohant4@bitbucket.org/amohant4/rramtraining.git
  ```