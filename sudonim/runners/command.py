from . import RUNNERS

def run_command( cmd, **kwargs ):
    """
    Invoke different commands like 'download', 'serve', 'bench'
    """
    runner = RUNNERS.get(cmd, None)

    if runner is None:
        raise ValueError(f"Unrecognized command: '{cmd}'")
    
    runner(**kwargs)
    
