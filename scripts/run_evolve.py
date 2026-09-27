from openevolve import evolve_code
import time
import sys
import os

BASE_PATH = os.getcwd()

sys.path.append(f"{BASE_PATH}/validators")
    
    
from evaluator import test 

start = time.time()

evolve_code(
    f"{BASE_PATH}/code/explicit_code.py",
    evaluator=test,
    iterations=10,
    config=f"{BASE_PATH}/evolve_config/config.yml",
    output_dir=f"{BASE_PATH}/output"
)

print(time.time() - start)