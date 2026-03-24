import subprocess
import os
import sys

# path to your Python interpreter in the venv
python_exe = ""

# path to config file
config_path = ""

# list of your scripts in order
scripts = [
    "1_Image_quality_check.py",
    "2_Batch_segmentation.py",
    "3_Presorting_LC.py"
]

scripts_dir = ""

for script in scripts:
    script_path = os.path.join(scripts_dir, script)
    print('=' * 60)
    print(f"Running {script}...")
    
    result = subprocess.run(
        [python_exe, script_path, config_path],
        check=True
    )

