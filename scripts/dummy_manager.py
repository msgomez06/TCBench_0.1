import subprocess, sys, os
from argparse import ArgumentParser

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.params_writers import write_several_seasons, write_one_year

# tracks_file = "/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/tracks/ibtracs/ibtracs.ALL.list.v04r00.csv"
# Old path: "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv"


def str2list(li):
    if type(li) == list:
        li2 = li
        return li2
    elif type(li) == str:

        if ", " in li:
            li2 = li[1:-1].split(", ")
        else:

            li2 = li[1:-1].split(",")
        return li2

    else:
        raise ValueError(
            "li argument must be a string or a list, not '{}'".format(type(li))
        )


parser = ArgumentParser()

parser.add_argument(
    "--seasons", type=str2list, help="input seasons, pass it as a list", default=[2000]
)
parser.add_argument(
    "--models",
    type=str2list,
    help="input models, pass it as a list",
    default=["graphcast"],
)
parser.add_argument(
    "--range",
    help="whether seasons should be interpreted as a range",
    action="store_true",
)
parser.add_argument(
    "--all_tcs",
    help="whether to use all tcs of the season or one at random per basin",
    action="store_true",
)
parser.add_argument(
    "--input",
    help="'cds' or 'local'. Note that fcnv can only be run with cds currently",
    default="local",
)
parser.add_argument(
    "--output",
    help="output path for input_params",
    default="/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/TCBench_0.1/input_params/",
)
parser.add_argument(
    "--slurm",
    help="path to slurm_manager.sh",
    default="/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/TCBench_0.1/slurms/slurm_manager.sh",
)
parser.add_argument(
    "--index",
    help="index of the job submission",
    default=None,
)

args = parser.parse_args()

print(args)
