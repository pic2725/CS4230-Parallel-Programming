# Getting started
## Building the assignment
```bash
git clone https://gitlab.chpc.utah.edu/01404975/assignment3a.git
cd assignment3a
# chpc has default read permissions for everyone, we set correct permission
bash init.sh
source env.sh
make
```
## Setting up environment
   On CHPC machines, use command `source env.sh` every time you log in
   to any machine (when you login to chpc and any time you run `srun` or
   `salloc`. For now `env.sh` sets gcc version and creates `env.log` file which
   you can share if you face problems.

## Correctness tests
   Makefile should create two executables for two problems. The executable don't take any command line arguement. To compile code, you don't need machine with GPU. To run the code, you need to allocate machine with gpu. Use command:
   ```bash
   srun -M notchpeak --account=notchpeak-gpu --partition=notchpeak-gpu --nodes=1 --ntasks=1 --gres=gpu:1  --pty /bin/bash -l
   ```
   > You can use other clusters e.g. kingspeak and you will need to change account+partition combination. Note the `--gres` option.
   > This option will make sure you will get the GPU for allocati11on.

## Advanced compilation options
   To get more performance, you can recompile your code after you get compute node (after `srun` gives you allocation). Your code
   will be then compiled for exact GPU installed in your compute node.

# Notes
   1. If you have questions, ask in piazza under `assignment3a` section.
