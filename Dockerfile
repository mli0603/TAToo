FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

RUN apt-get update 
RUN apt-get install ffmpeg git -y

RUN pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
RUN pip install mmsegmentation
RUN pip install git+https://github.com/princeton-vl/lietorch.git
RUN pip install h5py
RUN pip install einops
RUN pip install albumentations
RUN pip install future tensorboard
RUN pip install terminaltables
RUN pip install open3d
RUN pip install seaborn
RUN pip install natsort
RUN pip install pytransform3d
RUN pip install opt_einsum
RUN pip install wget
RUN pip install gdown

CMD ["bash"]
