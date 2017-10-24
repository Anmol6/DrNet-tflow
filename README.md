
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
### Example Usage:


#### Training DrNet
```
python run.py --num_gpus 1 --batch_size 50 --size_pose_embedding 5 --size_content_embedding 128 --max_steps 12 --num_epochs 2000 --run_name r1  
```

#### Training LSTM for Video generation

```
python run_lstm.py --num_gpus 1 --batch_size 40 --size_post_embedding 5 --size_content_embedding 128 --evaluate False --training True --num_epocs 10000 -- run_name r1_lstm --restore_dir_D /some/dir/D --restore_dir_Ep /some/dir/Ep --restore_dir_Ec /some/dir/Ec
```


Other hyperparameters are described in the ```run.py``` and ```run_lstm.py``` files. 




### Results
#### Training DrNet on KTH
Pose encoder Dimensions = 5
Content encoder Dimensions = 128
DCGAN Unet + DCGAN Pose Encoder

Decoder Loss 

![](https://user-images.githubusercontent.com/13502307/31972516-1fbadef6-b8ef-11e7-9566-ef9a4bde0927.png)


After 60k iterations:
Original frame on the left, frame to be decoded the middle, decoder output on the right
![](https://user-images.githubusercontent.com/13502307/31907767-c6f2c3f2-b802-11e7-8221-d79aec0c281b.png)



#### Training LSTM
Two 512 dimensional LSTMs, with tanh dense layer on top


L2 loss for predicted pose encodings:
![](https://user-images.githubusercontent.com/13502307/31972515-1fa773ac-b8ef-11e7-8b85-bc552cc787c2.png)

Example of some generated video frames:
![](https://user-images.githubusercontent.com/13502307/31972256-bebaee3a-b8ed-11e7-9320-0b728fa81ea7.png)
![](https://user-images.githubusercontent.com/13502307/31972257-bec6a9fa-b8ed-11e7-9151-fef03766eee2.png)





