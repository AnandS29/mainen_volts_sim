import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument(
  "--params",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=int,
  default=[1,1,1,1,1,1,1,1,1,1,1,1,1,1],  # default if nothing is provided
)
args = parser.parse_args()
np.savetxt("params512.csv", args.params)
