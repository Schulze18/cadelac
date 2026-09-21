# CaDeLaC

Source code for **Context-Aware Deep Lagrangian Networks for Model Predictive Control (CaDeLaC)**.

If you find this work useful, please consider citing:
```
@inproceedings{schulze2025contextawaredelan,
  author={Schulze, Lucas and Peters, Jan and Arenz, Oleg},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Context-Aware Deep Lagrangian Networks for Model Predictive Control}, 
  year={2025},
  volume={},
  number={},
  pages={6939-6946},
  keywords={},
  doi={10.1109/IROS60139.2025.11246292}
}
```

For experiment videos, check the [project website](https://schulze18.github.io/cadelac_website/).

# Installation - Training and Simulation

### Training and Simulation Setup
1. Clone the Repository
   ```bash
   git clone git@github.com:Schulze18/cadelac.git
   cd cadelac
   git submodule update --recursive --init
   ```

2. Set Up Conda Environment
   ```bash
   conda env create -f cadelac_env.yml
   conda activate cadelac
   ```

3. Install CaDeLaC as a Python pkg and additional Dependencies
   ```bash
   pip install -e .
   pip install l4casadi==1.4.1 --no-build-isolation
   ```


4. Install Acados (v0.4.3)
   Follow the [official installation guide](https://docs.acados.org/installation/):  
   ```bash
   cd acados
   mkdir -p build && cd build
   cmake -DACADOS_WITH_QPOASES=ON ..
   make install -j4
   ```

   Install the Python interface:  
   ```bash
   pip install -e acados/interfaces/acados_template
   ```

   Add the following lines to your `.bashrc` file:  
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"<acados_root>/lib"
   export ACADOS_SOURCE_DIR="<acados_root>"
   ```

---

### Real Robot Setup
1. Create a ROS Workspace
   ```bash
   mkdir -p ~/catkin_ws/src
   cd ~/catkin_ws/src
   git clone git@github.com:Schulze18/cadelac.git
   cd cadelac
   git submodule update --recursive --init
   ```


2. Set Up Conda with ROS
   ```bash
   conda env create -f cadelac_ros_env.yml
   conda activate cadelac_ros
   ```

   To ensure **Acados** and **Libfranka** use the same compiler, add the installed compiler (`gcc-12`) to your `PATH` (recommended in `.bashrc`):
   ```bash
   export CC=$HOME/miniconda3/envs/cadelac_ros/bin/x86_64-conda-linux-gnu-gcc
   export CXX=$HOME/miniconda3/envs/cadelac_ros/bin/x86_64-conda-linux-gnu-g++
   ```

3. **Install CaDeLaC and dependencies**
   ```bash
   pip install -e .
   pip install l4casadi==1.4.1 --no-build-isolation
   ```

4. **Install Acados (v0.4.3)**  
   Follow the step 4 from **Training and Simulation Setup**.


5. **Install Libfranka (v0.13.3)**  
   [Libfranka](https://github.com/frankaemika/libfranka) provides low-level control for Franka Emika research robots.

   ```bash
   git clone --recurse-submodules https://github.com/frankarobotics/libfranka.git
   cd libfranka
   git checkout 0.13.3
   git submodule update
   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/openrobots/lib/cmake -DBUILD_TESTS=OFF ..
   make
   ```

6. **Build the ROS workspace**
   ```bash
   cd ~/catkin_ws
   catkin_make -DPYTHON_EXECUTABLE=$(which python) -DCMAKE_BUILD_TYPE=Release -DFranka_DIR:PATH={PATH_TO_LIBFRANKA}/libfranka/build -j2
   source devel/setup.bash
   ```

## Datasets

The datasets used for training are available on [Hugging Face](https://huggingface.co/datasets/schulze18/cadelac):

```bash
git clone https://huggingface.co/datasets/schulze18/cadelac cadelac/learning/datasets/
```

## Training and Evaluation
All scripts related to training the proposed models are located in the `learning` folder.  
The training pipeline is adapted from [Deep Lagrangian Networks](https://github.com/milutter/deep_lagrangian_networks) and implemented in **PyTorch**.


### Evaluate Pretrained Model (IROS 2025)
```bash
python -m cadelac.learning.train_panda -l 2
```

### Learn Residual Dynamics
```bash
python -m cadelac.learning.train_panda -l 0
```
> **Note:**  
Due to the dataset size, data loading may take a few minutes. During training, each epoch took around 30s on an NVIDIA GeForce RTX 4080. In the paper, we trained for 3000 epochs, but 1000 epochs already achieve similar performance.

### Evaluate Trained Model
```bash
python -m cadelac.learning.train_panda -l 1
```

### Learn Robot Model with DeLaN
```bash
python -m cadelac.learning.train_panda -l 0 -f 1
```

### Evaluate DeLaN
```bash
python -m cadelac.learning.train_panda -l 1 -f 1
```

---

## Control Experiments

### Mujoco Simulation

To evaluate joint tracking under random loads:
```bash
python -m cadelac.control.eval_controllers_multiple_envs -c 2
```

Controller options:
- `-c 0`: Nominal MPC
- `-c 1`: MPC + EKF
- `-c 2`: CaDeLaC (Context-Aware MPC)

---

### Real Franka Robot

**Terminal 1:** Launch hardware interface
```bash
conda activate cadelac_ros
source catkin_ws/devel/setup.bas
roslaunch franka_example_controllers effort_joint_controller.launch robot_ip:={ROBOT_IP} load_gripper:=true robot:=panda
```

**Terminal 2:** Run CaDeLaC controller
```bash
conda activate cadelac_ros
source catkin_ws/devel/setup.bash 
cd catkin_ws/src/cadelac/
python -m cadelac.ros.cadelac_node -c 2
```

> **Note:**  
As Acados needs to compile the controller in the first run, which will take several minutes for CaDeLaC, the safety flag [compilation_run](https://github.com/Schulze18/cadelac/blob/main/cadelac/ros/cadelac_node.py#L63) prevents the controller to be executed and holds the robot in place. Once the compilation is done, you can stop the script, set the flag to `False` and run again the script which will load the compiled controller. If you run a new controller for the first time, set the flag to `True` again.

---
## TODO
- [ ] Additional implementation details
- [ ] Dataset collection scripts


# Acknowledgement
We gratefully acknowledge the support of NVIDIA Corporation through the NVIDIA Academic Grant Program, including the provision of RTX 6000 Ada Generation GPUs used in this research.
