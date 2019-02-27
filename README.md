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

   - The API folder: `/home/nianyi/models/research`
   - The project folder : `/home/nianyi/Documents/Object-Detection/BBox`
   
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
   
### SSH
To connect the server, you need to do 
```
ssh yournetID@lucia.duhs.duke.edu
```

### Docker
0. Install Docker-nvidia in Ubuntu
    - check the [procedure](https://blog.csdn.net/chxw098/article/details/79741586)
    - check the [procedure](https://chunml.github.io/ChunML.github.io/project/Installing-NVIDIA-Docker-On-Ubuntu-16.04/)
1. Install [TensorFlow image](https://www.tensorflow.org/install/docker)
    
    - Download TensorFlow Docker Image 

    `$ docker pull tensorflow/tensorflow:latest-devel-gpu-py3 `
    
    - Start a TensorFlow-configured container:
    
    `$ sudo docker run --runtime=nvidia -it tensorflow/tensorflow:latest-devel-gpu-py3 bash`
    
2. Install through Dockerfile
    - Go the project folder and run
    `docker build -t detection .`

    - Run Docker
    ```
    docker run --rm --runtime=nvidia -it -v `pwd`:/workspace -v /home/nianyi:/home/nianyi -v /media/Data:/media/Data detection bash
    ``` 
    - Close Docker
    Press `Ctrl + D`

    - Purging All Unused or Dangling Images, Containers, Volumes, and Networks
    `$ docker system prune`

    To additionally remove any stopped containers and all unused images (not just dangling images), add the -a flag to the command:

    `$ docker system prune -a`

    - Removing Docker Images

        - Remove one or more specific images
         Use the docker images command with the -a flag to locate the ID of the images you want to remove. This will show you every image, including intermediate image layers. When you've located the images you want to delete, you can pass their ID or tag to docker rmi:

        - List:

            `$ docker images -a`

        - Remove:

            `$ docker rmi Image Image`
   
### Data Cleaning   

1. Convert images to `png`
2. Image size issues:
   Q1: [Do I need to resize the images into the same size?](https://stackoverflow.com/questions/39334226/what-should-be-appropriate-image-size-input-to-faster-rcnn-caffe-model)
   A: Yes.
   
3. Crop Images and transfer the original labels to the new image size:
   - Crop and Pad images to the same size: `CropIm_GT.m`
   - Generate training, testing and validation set: `Generate_Train_Val_Test.m`
   
4. Transform images and labels into `record` format:
   - `Data_preprocessing_v0.py`: `python Data_preprocessing_v0.py train` and the output should be `train.record`
   - `python Data_preprocessing_v2.py train_v2`

### Train
1. Change dir to `/home/nianyi/models/research/object_detection` and run:
    ```
    python legacy/train.py --gpu 0 \
    --pipeline_config_path /home/maciej/Documents/Object-Detection/BBox/detection/faster_rcnn_resnet101_coco.config \
    --train_dir /home/nianyi/Documents/Object-Detection/BBox/detection/train/
    ```
    
### Train in Docker
1. Change dir to project folder: 
    ```
    cd /Documents/Object-Detection
    ```
2. Run docker container:
    ```
     sudo  docker run --rm --runtime=nvidia -it -v /home/maciej/Documents/Object-Detection:/workspace -v /home/maciej:/home/maciej detection bash
     ```
   To enabel `jupyter notebook` in docker image, run:
   ```
   sudo  docker run --rm --runtime=nvidia -p 10000:10000 -it -v /home/maciej/Documents/Object-Detection:/workspace -v /home/maciej:/home/maciej detection bash
   ```
   Or
   ```
   sudo  docker run --rm --runtime=nvidia -p 10000:10000 -it \
        -v /media/maciej/Nianyi/Object-Detection:/workspace \
        -v /home/maciej:/home/maciej detection bash
   ```   
   Or 
   ```
   sudo  docker run --rm --runtime=nvidia -p 8888:8888 -it -v /home/maciej/Documents/Object-Detection:/workspace -v /home/maciej:/home/maciej detection bash
   ```
   Note that, `-p 10000:10000` set the port to `10000`. Generally, people use `8888:8888` as the default setting.
3. Ensure that tensorflow is using GPU:
    ```
    import tensorflow as tf
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    ```
    Or, just using `tf.test.gpu_device_name()`

4. `Ctrl + D` to exit the docker or python
5. Change dir to `/tensorflow/models/research/object_detection` and run:
    ```
    python legacy/train.py --gpu 1 \
    --pipeline_config_path /home/maciej/Documents/Object-Detection/BBox/detection/faster_rcnn_resnet101_coco.config \
    --train_dir /home/maciej/Documents/Object-Detection/BBox/detection/train/
    ```
    Or
    ```
    python /tensorflow/models/research/object_detection/legacy/train.py --gpu 1 \
    --pipeline_config_path /home/maciej/Documents/Object-Detection/BBox/detection/faster_rcnn_resnet101_coco.config \
    --train_dir /home/maciej/Documents/Object-Detection/BBox/detection/train/
    ```   
    Or
    ```
    python /tensorflow/models/research/object_detection/legacy/train.py --gpu 1 \
    --pipeline_config_path /home/maciej/Documents/Object-Detection/BBox/detection/faster_rcnn_resnet101_coco_V1.config \
    --train_dir /home/maciej/Documents/Object-Detection/BBox/detection/train_V1/
    ``` 
    Or
    ```
    python /tensorflow/models/research/object_detection/legacy/train.py --gpu 1 \
    --pipeline_config_path /workspace/BBox/detection/faster_rcnn_resnet101_coco_V2.config \
    --train_dir /workspace/BBox/detection/train_V2/
    ```   
    
 6. Cross Evaluation
    ```
    python /tensorflow/models/research/object_detection/legacy/eval.py --gpu 0 \
        --checkpoint_dir /home/maciej/Documents/Object-Detection/BBox/detection/train_V1/ \
        --eval_dir /home/maciej/Documents/Object-Detection/BBox/detection/eval_V1/ \
        --pipeline_config_path /home/maciej/Documents/Object-Detection/BBox/detection/faster_rcnn_resnet101_coco_V1.config
    ```
    Or
    ```
        python legacy/eval.py --checkpoint_dir /home/maciej/Documents/Object-Detection/BBox/detection/train_V1/ \
        --eval_dir /home/maciej/Documents/Object-Detection/BBox/detection/eval_V1/ \
        --pipeline_config_path /home/maciej/Documents/Object-Detection/BBox/detection/faster_rcnn_resnet101_coco_V1.config
    ```
    Or
    ```
    python /tensorflow/models/research/object_detection/legacy/eval.py --gpu 0 \
        --checkpoint_dir /workspace/BBox/detection/train_V2/ \
        --eval_dir /workspace/BBox/detection/eval_V2/ \
        --pipeline_config_path /workspace/BBox/detection/faster_rcnn_resnet101_coco_V2.config
    ```
    
  7. Export inference model
  - Generate [`pb`](https://blog.csdn.net/qq_34106574/article/details/80151574) file  
     ```
     python /tensorflow/models/research/object_detection/export_inference_graph.py --input_type image_tensor \
        --pipeline_config_path /home/maciej/Documents/Object-Detection/BBox/detection/faster_rcnn_resnet101_coco_V1.config \
        --trained_checkpoint_prefix /home/maciej/Documents/Object-Detection/BBox/detection/train_V1/model.ckpt-17876 \
        --output_directory /home/maciej/Documents/Object-Detection/BBox/detection/inference_V1/
     ```
     Or
     ```
     python /tensorflow/models/research/object_detection/export_inference_graph.py --input_type image_tensor \
        --pipeline_config_path /workspace/BBox/detection/faster_rcnn_resnet101_coco_V2.config \
        --trained_checkpoint_prefix /workspace/BBox/detection/train_V2/model.ckpt-20000 \
        --output_directory /workspace/BBox/detection/inference_V2/
     ```
  - In the `ipynb` folder, run inference in `jupyter`
    ```
    jupyter notebook --ip 0.0.0.0 --port 10000 --no-browser --allow-root
    ```
    Note that, to access Jupyter notebook running on Docker container, you need to run your notebook on `0.0.0.0`
    
  8. Tensorboard
        ```
        tensorboard --logdir=/home/maciej/Documents/Object-Detection/BBox/detection/eval/
        ```
    
  9.  Inference (Getting the bbox and detection images)
  - In the `detection` folder, run 
    ```
    python inference_v1.py
    ```
    Or
    ```
    python inference_v2.py
    ```
   
