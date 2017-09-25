
# drnet-tflow
Tensorflow implementation of DrNet

ssh ... Build the Docker image with
```
$ docker build --no-cache -t img-dockername-video . 
```

Build the docker container:

```
$ NV_GPU=0,2 nvidia-docker run -it -v ~/projects/drnet-tflow/:/drnet-tflow/  -v /mnt/:/mnt/ -p 8894:8888 --name container-name img-dockername-video bash

```
Example Usage:

```
python train.py --num_gpus 8 --batch_size 50 --size_pose_embedding 5 --size_content_embedding 128 --max_steps 12 --num_epochs 2000 --run_name r1  
```

Other hyperparameters are described in the ```train.py``` file. 
