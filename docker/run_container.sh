#!/bin/bash

# Run the Docker container - NOTE: This must be ONE command, no line breaks!
docker run -itd \
     --name noetic \
     --net=host \
     --privileged \
     -e DISPLAY=$DISPLAY \
     -e QT_X11_NO_MITSHM=1 \
     -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
     -v "$(pwd)/..":/ros2_ws/src \
     ros1_noetic bash