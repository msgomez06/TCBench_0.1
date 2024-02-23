from argparse import ArgumentParser
from utils.main_utils import str2list
from case_study_plot import trajectory_no_pp, trajectory_with_pp

func_dict = {
    "trajectory_no_pp": trajectory_no_pp,
    "trajectory_with_pp": trajectory_with_pp
}
parser = ArgumentParser()


# data paths

parser.add_argument("--data_path", type=str, default="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/", help="Path to the data folder.")
parser.add_argument("--df_path", default="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv")
parser.add_argument("plot_path", default="/users/lpoulain/louis/plots/case_studies/")


# function to use

parser.add_argument("--func", type=str, default="trajectory_no_pp", help="Function to use for the case study.", choices=func_dict.keys())

# function arguments

parser.add_argument("--tc_id", type=str, default="2005236N23285", help="ID of the tropical cyclone to plot.", choices=["2005236N23285", "2000185N15117"])
parser.add_argument("--model_names", type=str2list, default=["panguweather"], choices=["panguweather", "graphcast", "fourcastnetv2"])
parser.add_argument("--max_lead", type=int, default=72, help="Maximum lead time to plot.")

# specific arguments for pp plot

parser.add_argument("--pp_type", type=str, default="linear", help="Type of post-processing to use.", choices=["linear", "xgboost", "cnn"])
parser.add_argumnet("--train_seasons", type=str2list, default=["2005"])
parser.add_argument("--basin", type=str, default="NA", help="Basin of the tropical cyclone.", choices=["NA", "WP", "EP", "NI", "SI", "SP", "SA"])

## for linear pp
parser.add_argument("--dim", type=int, default=2, help="Dimension of the linear post-processing.", choices=[1,2])

## for xgboost pp
parser.add_argument("--lr", type=float, default=0.05, help="Learning rate for the xgboost model.")
parser.add_argument("--depth", type=int, default=3, help="Depth of the trees for the xgboost model.")
parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for the xgboost model.")
parser.add_argument("--gamma", type=float, default=0.0, help="Gamma for the xgboost model.")
parser.add_argument("--jsdiv", action="store_true", help="Use Jensen-Shannon divergence as loss function")
parser.add_argument("--sched", action="store_true", help="Use a scheduler for the learning rate [cos_annealing]")
parser.add_argument("--stats", type=str2list, default=[], help="Statistics to use for the xgboost post-processing.")
parser.add_argument("--stats_wind", type=str2list, default=["max"], help="Statistics to use for the wind in the xgboost post-processing.")
parser.add_argument("--stats_pres", type=str2list, default=["min"], help="Statistics to use for the pressure in the xgboost post-processing.")


args = parser.parse_args()

pp_params = {"dim": args.dim,
             "jsdiv": args.jsdiv,
             "sched": args.sched,
             "stats": args.stats,
             "stats_wind": args.stats_wind,
             "stats_pres": args.stats_pres,
             "lr": args.lr,
             "depth": args.depth,
             "epochs": args.epochs,
             "gamma": args.gamma,
             "train_seasons": args.train_seasons,
             "basin": args.basin}

tc_id = args.tc_id
model_names = args.model_names
max_lead = args.max_lead

pp_type = args.pp_type

func_dict[args.func](tc_id=tc_id, model_names=model_names, max_lead_time=max_lead, pp_type=pp_type, pp_params=pp_params, 
                     data_path=args.data_path, df_path=args.df_path, plot_path=args.plot_path)
