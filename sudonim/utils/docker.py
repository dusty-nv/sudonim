# Some basic utilities for finding/starting/stopping containers
# These require 'pip install docker' and the docker socket to be mounted: 
#   -v /var/run/docker.sock:/var/run/docker.sock
import sudonim

env, log = sudonim.getenv()

class Docker:
    """
    Some basic utilities for starting/stopping containers
    This requires the docker socket to be mounted:
      /var/run/docker.sock:/var/run/docker.sock
    """
    Client = None

    @staticmethod
    def client():
        if not env.HAS_DOCKER_API:
            raise ImportError(f"Attempted to use a docker function without having docker installed ('pip install docker')")
        import docker
        if not Docker.Client:
            Docker.Client = docker.from_env()
        return Docker.Client
    
    @staticmethod
    def find(names):
        if isinstance(names, str):
            names=[names]
        try:
            for c in Docker.client().containers.list():
                for name in names:
                    if name in c.name:
                        return c.name
            log.warning(f"Failed to find container by the names {names}")
        except Exception as error:
            log.error(f"Exception trying to find container {names} ({error})")

    @staticmethod
    def stop(name):
        try:
            name = Docker.find(name)
            if name:
                c = Docker.client().containers.get(name)
                log.info(f"Stopping container '{c.name}' ({c.id})")
                c.stop()
        except Exception as error:
            log.error(f"Failed to stop container '{name}' ({error})")
            Docker.kill(name)
        
    @staticmethod
    def kill(name):
        name = Docker.find(name)
        if name:
            c = Docker.client().containers.get(name)
            log.info(f"Killing container '{c.name}' ({c.id})")
            c.kill()