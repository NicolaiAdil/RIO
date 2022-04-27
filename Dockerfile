FROM ros:melodic-ros-core-bionic

SHELL ["/bin/bash", "-c"] 

#Required folder for Ladybug SDK
RUN mkdir -p /Documents

# Auto source ros
#RUN echo 'source /opt/ros/melodic/setup.bash' >> /home/revolt/.bashrc

## Add repo for GDAL binaries, and update repo information
# RUN echo "deb http://ppa.launchpad.net/ubuntugis/ppa/ubuntu bionic main" >> /etc/apt/sources.list
# RUN echo "deb-src http://ppa.launchpad.net/ubuntugis/ppa/ubuntu bionic main" >> /etc/apt/sources.list
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 089EBE08314DF160

# # GDAL paths needed for pip installation
# ENV CPLUS_INCLUDE_PATH /usr/include/gdal
# ENV C_INCLUDE_PATH /usr/include/gdal

# OpenCV path, needed for catkin make
ENV OpenCV_DIR /usr/share/OpenCV

# Python pip installs are much slower than apt, installing these first ensures
# that we only have to rerun these when the python requirements change.

# Apt pacakges required for the python pip install (Mostly GDAL stuff)
RUN apt-get -qq update --fix-missing && \
    apt-get -qq install -y --allow-unauthenticated \
    curl \
    python-dev \
    python3-dev \
    python-catkin-tools \
    python-opencv \
    python-numpy=1:1.13.3-2ubuntu1 \
    python-scipy=0.19.1-2ubuntu1 \
    python-pymodbus=1.3.2-1 \
    python-pyasn1=0.4.2-3 \
    python-twisted-conch=1:17.9.0-2ubuntu0.3 \
    python-tk=2.7.15~rc1-1

# Both of these next packages are required for GDAL pip install.
# They need to be kept in to be in a separate RUNs!
# RUN apt-get -qq update --fix-missing && \
#     apt-get -qq install -y --allow-unauthenticated \
#     libogdi3.2=3.2.0+ds-2 

# RUN apt-get -qq update --fix-missing && \
#     apt-get -qq install -y --allow-unauthenticated \
#     ros-melodic-pcl-conversions 
    
# Installing and upgrading pip. We use the same pip version for python3 and python2
# RUN curl https://bootstrap.pypa.io/pip/2.7/get-pip.py -o get-pip.py
# RUN python2 get-pip.py
# RUN python3 get-pip.py

# Installing pip dependencies
# COPY requirements.txt /home/revolt/requirements.txt
# COPY requirements3.txt /home/revolt/requirements3.txt

# # The --ignore-installed flag can be explained here: https://github.com/pypa/pip/issues/5247
# RUN python2 -m pip install -r /home/revolt/requirements.txt \
#     --ignore-installed pyasn1-modules
# RUN python3 -m pip install -r /home/revolt/requirements3.txt

# Other non-python apt packages
RUN apt-get -qq update --fix-missing && \
    apt-get -qq install -y --allow-unauthenticated \
    # Utility 
    # iputils-ping \
    # pciutils \
    # usbutils \
    # dos2unix \
    # openssh-server \
    # nano \
    # vim \
    # net-tools \
    # Various libraries
    libpcap-dev=1.8.1-6ubuntu1.18.04.1 \
    libyaml-cpp-dev=0.5.2-4ubuntu1 \
    # libdxflib-dev \
    # openjdk-8-jdk \
    # openjdk-8-jre \
    # Note: dbus-x11 required to run GUI applications
    dbus-x11 \
    # Install ladybug dependencies
    # libxerces-c3.2 \
    # libraw1394-11 \
    # libc6 \
    # libusb-1.0-0 \
    # wget \
    libavcodec57 \
    libavformat57 \
    libavutil55 \
    # libswscale4 \
    # libglu1-mesa \
    # libomp5 \
    # ROS specifics
    # ros-melodic-rosserial \
    # ros-melodic-rosserial-arduino \
    # ros-melodic-diagnostic-updater \ 
    ros-melodic-roslint \
    # ros-melodic-gps-common \
    # ros-melodic-geographic-msgs \
    ros-melodic-tf-conversions \
    ros-melodic-tf2-kdl \
    ros-melodic-dynamic-reconfigure \
    ros-melodic-pcl-ros \
    ros-melodic-laser-geometry \
    # ros-melodic-nmea-msgs \
    # ros-melodic-nmea-comms \
    ros-melodic-tf2-geometry-msgs \
    ros-melodic-image-transport \
    ros-melodic-rqt-reconfigure \
    ros-melodic-nav-msgs \
    ros-melodic-map-server \ 
    ros-melodic-costmap-2d \ 
    ros-melodic-velodyne \
    ros-melodic-message-filters \
    # ros-melodic-rviz \
    ros-melodic-tf2-geometry-msgs \ 
    ros-melodic-image-geometry \
    ros-melodic-vision-opencv\
    ros-melodic-cv-bridge \
    ros-melodic-jsk-rviz-plugins \
    ros-melodic-rosbridge-server

# Old Ladybug SDK installation steps that may be of use later
# RUN apt-get install xsdcxx --assume-yes
# RUN apt-get --reinstall install grub-pc --assume-yes
# RUN apt-get install grub --assume-yes
# RUN sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"/GRUB_CMDLINE_LINUX_DEFAULT="quiet splash usbcore.usbfs_memory_mb=1000"/g' /etc/default/grub
# #RUN update-grub

# Start dbus service to allow GUI applications to run
RUN service dbus start

COPY . /sensor_fusion_ws
RUN source /opt/ros/melodic/setup.bash && cd /sensor_fusion_ws && catkin build

COPY ./entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]