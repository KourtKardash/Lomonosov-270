import subprocess
import os

model = "code/nn.py"
base_cmd = "python3"

container_image = "/scratch/s02210430/test/noisy-cells.sqsh"

log_dir = "/scratch/s02210430/test/logs"
os.makedirs(log_dir, exist_ok=True)

log_files = [
    os.path.join(log_dir, "log_1.txt"),
    os.path.join(log_dir, "log_2.txt"),
    os.path.join(log_dir, "log_3.txt")
]

processes = []
for idx in range(3):
    cmd = f"{base_cmd} {model}"

    srun_cmd = f"srun --gres=gpu:1 --mem=10G --container-image {container_image} bash -c '{cmd}'"

    with open(log_files[idx], "w") as log:
        process = subprocess.Popen(srun_cmd, shell=True, stdout=log, stderr=log)
        processes.append(process)

for process in processes:
    process.wait()
