# TensorFlow GPU setup

There are many tutorials out there explaining how to configure
a host machine for running GPU-accelerated TensorFlow. Below
I describe what I found to be the most hassle-free path.

First, install the NVidia kernel drivers. The steps depend on your
Linux distribution. It's important that `nvidia-smi` tool runs
and detects your GPU.

Next step is to install all the necessary userspace components.
This can be a daunting task. You can instead download an already
setup image and run it in a container. Use the definition in this
directory to build it:

	singularity build tfgpu.sif tfgpu.def

After that, run the container when a TensorFlow program must be executed:

	singularity run --bind /mnt/nvme:/mnt/nvme --nv tfgpu.sif

Notice how any directory on the host can be bind-mounted and thus made
visible to the container runtime.

Once inside the container runtime, ensure that it can utilize the GPU:

	./ml/check-tf-gpu.py
