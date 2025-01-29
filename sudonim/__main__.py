import sys
import time
import datetime
import sudonim as nim

env, log = nim.getenv()

def main():
    args = nim.ArgParser().parse_args()

    header = nim.property_table(
        env, filter=nim.filter_env, attrs=['reverse'],  # 'dark'
        wrap_rows=5, merge_columns=True,
    )

    log.success(f'sudonim version {nim.__version__}\n\n{header}\n')

    if args.version:
        sys.exit()

    time_begin = time.perf_counter()

    for cmd in args.commands:
        nim.run_command(cmd, **vars(args))

    time_elapsed = datetime.timedelta(seconds=time.perf_counter()-time_begin)
    log.success(f'sudonim - shutting down, uptime {time_elapsed}')

if __name__ == "__main__":
  main()
