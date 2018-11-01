# Object-Detection-FRCNN

## Reproduction steps

### Prerequisites

1. Install Tensorflow 1.9.0 [through virtualenv](https://www.tensorflow.org/install/pip)

    ```
    # For CPU
     (venv) $ pip install tensorflow == 1.9.0
    # For GPU
     (venv) $ pip install tensorflow-gpu == 1.9.0
    ```
    - Activate `virtualenv`: `$ source ./venv/bin/activate`
    
    - Deactivate `virtualenv`: `(venv) $ deactivate`
    
    Show Tensorflow installation information
    - `(venv) $ pip show tensorflow`
    
    Which returns
    ```
    Name: tensorflow
    Version: 1.9.0
    Summary: TensorFlow is an open source machine learning framework for everyone.
    Home-page: https://www.tensorflow.org/
    Author: Google Inc.
    Author-email: opensource@google.com
    License: Apache 2.0
    Location: /home/nianyi/venv/lib/python3.5/site-packages
    Requires: numpy, grpcio, astor, protobuf, setuptools, absl-py, gast, wheel, tensorboard, six, termcolor
    ```
2. Install Tensorflow object detection API as described in the installation section of
[object_detection](https://github.com/tensorflow/models/tree/master/research/object_detection)

   The API folder: `/home/nianyi/models/research`
   
   Add `PYTHONPATH` to `virtualenv` environment:
   You can add the following line to your virtualenv's bin/activate file:
   `export PYTHONPATH="/the/path/you/want"`
   
   To open `jupyter` in [`virtualenv`](https://anbasile.github.io/programming/2017/06/25/jupyter-venv/):
   ```
   # Inside this folder create a new virtual environment:
   python -m venv projectname
   # Then activate it:
   source projectname/bin/activate
   # (Can Skip) From inside the environment install ipykernel using 'pip':
   pip install ipykernel
   # Install a new kernel:
   ipython kernel install --user --name=projectname
   # Open Jupyter in the projectfolder:
   jupyter notebook
   ```
   tl;dr
   ```
   $ python -m venv projectname
   $ source projectname/bin/activate
   (venv) $ pip install ipykernel
   (venv) $ ipython kernel install --user --name=projectname
   (venv) $ jupyter notebook
   ```
   To terminate `Jupyter`, you need do it from the website. Simply close the tab cannot terminate the project.
   
### Data Cleaning   

1. Convert images to `png`
2. Image size issues:
   Q1: [Do I need to resize the images into the same size?](https://stackoverflow.com/questions/39334226/what-should-be-appropriate-image-size-input-to-faster-rcnn-caffe-model)
   A: Yes.
   
3. 
   
   
