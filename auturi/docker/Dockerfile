FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# RUN apt-get update
RUN pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

COPY /home/ooffordable/ray_fork/ /ray_fork
