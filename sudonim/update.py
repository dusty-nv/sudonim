import os
import runpy
import subprocess

from pathlib import Path
from termcolor import cprint

SUDONIM_UPDATE = os.environ.get(
  'SUDONIM_UPDATE', os.environ.get(
    'DOCKER_PULL', False
))

SUDONIM_REPO = os.environ.get(
  'SUDONIM_REPO',
  'https://github.com/dusty-nv/sudonim'  # @branch
)

def auto_update(enabled=SUDONIM_UPDATE, run_module='sudonim', **kwargs):
    enabled = str(enabled).lower()
    if enabled in ['true', 'on' 'yes', '1', 'always', 'auto']:
        try:
            cmd = 'git pull'
            cwd = Path(__file__).parents[1]
            cprint(f"\nRunning auto update command '{cmd}' in {cwd}\n", attrs=['dark'])
            result = subprocess.run(
                cmd, cwd=cwd, executable='/bin/bash', shell=True, check=True, 
            )
        except Exception as error:
            cprint(f"\n{error} (ignoring failed update)\n", attrs=['dark'])

    if run_module:
        runpy.run_module('sudonim', run_name='__main__')
