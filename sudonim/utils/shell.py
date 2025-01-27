import subprocess

from sudonim import getLogger
from termcolor import colored

log = getLogger()

def shell(cmd, echo=True, capture_output=False, **kwargs):
    """ 
    Run shell command and return the result 
    """
    if not isinstance(cmd, list):
        cmd = [cmd]
        
    cmd = [x for x in cmd if x != None and len(x) > 0]
    
    if echo:
        endline = f' \\\n    '
        echo = echo if isinstance(echo, str) else 'Running shell command'
        log.info(f"{echo}:\n\n  {colored(endline.join(cmd), 'green')}\n")

    kwargs.setdefault('executable', '/bin/bash')
    kwargs.setdefault('shell', True)
    kwargs.setdefault('check', True)
    kwargs.setdefault('capture_output', capture_output)
    kwargs.setdefault('text', capture_output)

    return subprocess.run(' '.join(cmd), **kwargs)

def subshell(cmd, capture_output=True, **kwargs):
    """ 
    Run a shell and capture the output by default
    """
    return shell(cmd, capture_output=capture_output, **kwargs).stdout