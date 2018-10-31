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
    - Activate `virtualenv`: `(venv) $ source ./venv/bin/activate`
    
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

   The API folder: `/home/nianyi/venv/lib/python3.5/site-packages/tensorflow/models/research`
   
   Add `PYTHONPATH` to `virtualenv` environment:
   You can add the following line to your virtualenv's bin/activate file:
   `export PYTHONPATH="/the/path/you/want"`
   
   To open jupter in `virtualenv`:
   ```
   
