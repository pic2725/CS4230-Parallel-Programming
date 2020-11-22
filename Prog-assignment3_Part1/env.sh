module load gcc/6.4.0.lua
module load cuda/10.0.lua
echo Host: $USER@$(hostname) | tee env.log
echo CPU: $(lscpu | grep "Model name" | cut -d: -f 2 | awk '{$1=$1};1') | tee -a env.log
echo GCC ver: $(gcc -v 2>&1 | grep version | cut -d\  -f 3) | tee -a env.log
echo Job ID: $SLURM_JOB_ID | tee -a env.log
echo Num cores: $SLURM_CPUS_PER_TASK | tee -a env.log
python cuda_check.py | tee -a env.log
nvidia-smi | tee -a env.log
