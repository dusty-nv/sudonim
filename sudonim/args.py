import sys
import time
import datetime
import argparse
import pprint

from sudonim import RUNNERS, QUANTIZATIONS, run_command, getenv, filter_env, format_table, __version__

env, log = getenv()

class ArgParser(argparse.ArgumentParser):
    """ 
    Add command-line options and some post-parsing actions to the argument parser.
    """
    def __init__(self, **kwargs):
        """
        Populate an ``argparse.ArgumentParser`` with additional options for configuration.
        """
        kwargs.setdefault('formatter_class', argparse.ArgumentDefaultsHelpFormatter)
        kwargs.setdefault('description', f"Local AI microservices launcher for downloading, quantizing, and serving multimodal models.")
        
        super().__init__(**kwargs)
        
        self.add_argument('commands', type=str, choices=RUNNERS, nargs='*', default='serve', help="Run a sequence of these commands for downloading and serving models")

        grp = self.add_argument_group('API', description="Selects which API to use and sets authentication tokens")

        grp.add_argument('--api', type=str, default=None, choices=['mlc', 'trt_llm', 'vllm', 'hf', 'llama_cpp'], help="Inference API - will attempt to determine from the model type + quantization")
        grp.add_argument('--api-key', type=str, default=env.HF_TOKEN, metavar='$HF_TOKEN', help="Personal access token for gated/private models (inherits $HF_TOKEN or $NGC_API_KEY)")

        grp = self.add_argument_group('MODEL', description="Specify paths to local/remote models and their settings")

        grp.add_argument('--model', type=str, default=None, metavar='PATH', help="URL or local path to the model. This can be a repo ID on HuggingFace")
        grp.add_argument('--dataset', type=str, default=None, metavar='PATH', help="URL or path to datasets for when they are used, like for downloads or benchmarking")
        grp.add_argument('--tokenizer', type=str, default=None, metavar='PATH', help="URL or path of tokenizer (used by the benchmark client when needed)")

        grp.add_argument('--quantization', type=str, default='q4f16_ft', metavar='QUANT', help=f"Type of quantization to apply to the model:\n{pprint.pformat(QUANTIZATIONS, indent=2)}") #, choices=['q4f16_ft', 'q4f16_1'])
        grp.add_argument('--chat-template', type=str, default=None, metavar='CHAT', help="Manually specify or overide the model's chat template.")

        grp.add_argument('--max-context-len', type=int, default=None, help="Max number of tokens in the context, including prompt + reply")
        grp.add_argument('--max-batch-size', type=int, default=None, help="Max batch size for serving parallel requests")
        grp.add_argument('--prefill-chunk', type=int, default=None, help="Max number of tokens that can prefilled into the KV cache at once")

        grp = self.add_argument_group('SERVER', description="Networking configuration of the endpoint server.")
        grp.add_argument('--host', type=str, default='0.0.0.0', help="IP address or hostname of the local endpoint server, or its interfaces to bind to (0.0.0.0)")
        grp.add_argument('--port', type=int, default=9000, help="Port of the local endpoint server")

        grp = self.add_argument_group('CACHES', description="Sets various mounting locations on the server's host filesystem used to store data and models.")
        grp.add_argument('--cache-root', type=str, default=env.CACHE_ROOT, metavar='DIR', help=f'Default top-level mounted cache directory')

        for cache, cache_dir in env.CACHES.items():
            grp.add_argument(f'--cache-{cache}', type=str, default=cache_dir, metavar='DIR', help=f'Mount to store data for {cache.upper()}')
        
        grp = self.add_argument_group('LOGGING', description="Controls logging settings and verbosity levels")
                                         
        grp.add_argument('--version', action='store_true', help='Print system/environment info')
        grp.add_argument('--debug', '--verbose', action='store_true', help="Set the logging level to 'debug'")
        grp.add_argument('--log-level', default=None, type=str, choices=['debug', 'info', 'warning', 'error', 'critical'], help="Set the logging level")

    def parse_args(self, **kwargs):
        """
        Override for parse_args() that does some additional configuration
        """
        args = super().parse_args(**kwargs)
        
        if len(args.commands) == 0:
            args.commands = ['serve']

        if args.debug:
            args.log_level = 'debug'

        log.basicConfig(level=args.log_level)
        log.debug(f"{args}")
        
        print('')
        return args