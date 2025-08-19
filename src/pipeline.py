import subprocess, sys

steps = [
    ['python','src/data_prep.py'],
    ['python','src/train.py'],
    ['python','src/evaluate.py']
]

for cmd in steps:
    print('>>', ' '.join(cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        sys.exit(res.returncode)
print('Pipeline finished.')
