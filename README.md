# VizFlyt

## Installation: 

### Install Pytorch

```bash
conda create --name nerfstudio2 -y python=3.8

conda activate nerfstudio2

python -m pip install --upgrade pip

pip install pyyaml typeguard

pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Install Nerfstudio from Source

```bash
git clone https://github.com/nerfstudio-project/nerfstudio.git

cd nerfstudio

pip install --upgrade pip setuptools

pip install -e .
```


## DATA GENERATION AND SPLAT TRAINING

```bash
# Training
python nerfstudio/scripts/train.py splatfacto --data ./data/washburn-env6-itr0-1fps_nf_format/ --output-dir outputs/washburn-env6-itr0-1fps



# Viewer 
python nerfstudio/scripts/viewer/run_viewer.py --load-config ./outputs/forest_28thfeb_400/forest_28feb_400_nf_format/splatfacto/2024-02-28_185326/config.yml

# python scripts/viewer/run_viewer.py --load-config ./outputs/washburn-env6-itr0-1fps/washburn-env6-itr0-1fps_nf_format/splatfacto/2025-02-13_191419/config.yml

# Some additional installations 
pip install empy==3.3.4
pip install catkin_pkg 
pip install lark

source_ws -> source install/setup.bash
build_ws -> colcon build --symlink-install


# creating a new package
ros2 pkg create --build-type ament_python --node-name my_node my_package


# Launch the Vizflyt Launch file 
ros2 launch vizflyt vizflyt.launch.py 
```

# VizFlyt Viewer changes
 

# Running using ROS2

```bash
ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool "{value: true}"

ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode "{custom_mode: 'OFFBOARD'}"

ros2 service call /mavros/cmd/takeoff mavros_msgs/srv/CommandTOL "{altitude: 5.0}"

ros2 topic pub --rate 20 /mavros/setpoint_velocity/cmd_vel geometry_msgs/msg/TwistStamped "{twist: {linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {z: 0.0}}}"

ros2 topic pub --rate 10 /mavros/setpoint_velocity/cmd_vel geometry_msgs/msg/TwistStamped "{twist: {linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {z: 0.0}}}"

ros2 service call /mavros/cmd/land mavros_msgs/srv/CommandTOL "{}"

```































































