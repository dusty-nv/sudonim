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
    """
    Pull / install the latest version from github when DOCKER_PULL or SUDONIM_UPDATE is set.
    This is 'sudonim' shell script entrypoint and is able to reload the module after updates.
    """
    enabled = str(enabled).lower()
    if enabled in ['1', 'true', 'on', 'yes', 'y', 'enable', 'enabled', 'always', 'auto']:
        try:
            cmd = 'git pull && pip3 install --upgrade-strategy only-if-needed -e .'
            cwd = Path(__file__).parents[1]
            cprint(f"\nRunning auto update command in {cwd}\n  {cmd}", attrs=['dark'])
            result = subprocess.run(
                cmd, cwd=cwd, executable='/bin/bash', shell=True, check=True, 
            )
        except Exception as error:
            cprint(f"\n{error}\n", attrs=['dark'])

    if run_module:
        runpy.run_module('sudonim', run_name='__main__')
