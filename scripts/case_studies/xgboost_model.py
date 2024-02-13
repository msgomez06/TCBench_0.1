import numpy as np
import xgboost as xgb
import sys
from argparse import ArgumentParser
from utils.main_utils import str2list
from xgboost_utils import create_dataset, create_booster, train_save_xgb, plot_xgb, cos_annealing




def gbtree(data_path, model_name, lead_time, seasons, split_ratio:list,
            max_depth=6, n_rounds=30, learning_rate=0.3, gamma=0., stats=[],
            stats_wind=["max"], stats_pres=["min"],
            df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
            save_path="/users/lpoulain/louis/plots/xgboost/"
            ):
    
    model_name = "pangu" if model_name=="panguweather" else model_name
    dtrain_wind, dval_wind, dtest_wind,\
    dtrain_pres, dval_pres, dtest_pres,\
    stats, stats_wind, stats_pres, nb_tc,\
    truth_wind_test, truth_pres_test = create_dataset(data_path, model_name, lead_time, seasons, split_ratio,
                                            stats=stats, stats_wind=stats_wind, stats_pres=stats_pres,
                                            df_path=df_path)
    
    params, evallist_wind, evallist_pres = create_booster("gbtree", dtrain_wind, dval_wind, dtrain_pres, dval_pres, max_depth, learning_rate, gamma)
    
    # train and save models
    
    wind_save = f"xgb_wind_{model_name}_{lead_time}h_{'_'.join(s for s in seasons)}_depth_{max_depth}_epoch_{n_rounds}_lr_{learning_rate}_g_{gamma}" +\
                        f"_{'_'.join(stat for stat in stats_wind)}.json"
    pres_save = f"xgb_pres_{model_name}_{lead_time}h_{'_'.join(s for s in seasons)}_depth_{max_depth}_epoch_{n_rounds}_lr_{learning_rate}_g_{gamma}" +\
                        f"_{'_'.join(stat for stat in stats_pres)}.json"
    
    scheduler_fct = lambda x: cos_annealing(x, n_rounds, learning_rate)                    
    callback = xgb.callback.LearningRateScheduler(scheduler_fct)
                        
    bst_wind, eval_res_wind = train_save_xgb(params, dtrain_wind, n_rounds, evallist_wind, early_stopping_rounds=n_rounds,
                                            save_path=save_path, save_name=wind_save, callbacks=[callback])
    bst_pres, eval_res_pres = train_save_xgb(params, dtrain_pres, n_rounds, evallist_pres, early_stopping_rounds=n_rounds,
                                            save_path=save_path, save_name=pres_save, callbacks=[callback])
    
    # predict and plot
    
    ypred_wind = bst_wind.predict(dtest_wind, iteration_range=(0, bst_wind.best_iteration + 1))
    ypred_pres = bst_pres.predict(dtest_pres, iteration_range=(0, bst_pres.best_iteration + 1))
    
    test_loss_wind = np.sqrt(np.mean((ypred_wind - truth_wind_test)**2))
    test_loss_pres = np.sqrt(np.mean((ypred_pres - truth_pres_test)**2))

    
    plot_xgb(bst_wind=bst_wind, bst_pres=bst_pres, eval_res_wind=eval_res_wind, eval_res_pres=eval_res_pres, 
             model_name=model_name, xgb_model="gbtree", lead_time=lead_time, seasons=seasons, nb_tc=nb_tc,
             max_depth=max_depth, n_rounds=n_rounds, learning_rate=learning_rate, gamma=gamma, stats=stats, 
             stats_wind=stats_wind, stats_pres=stats_pres, test_loss_wind=test_loss_wind, test_loss_pres=test_loss_pres,
             truth_wind_test=truth_wind_test, truth_pres_test=truth_pres_test, y_pred_wind=ypred_wind, y_pred_pres=ypred_pres,
             save_path="/users/lpoulain/louis/plots/xgboost/"
             )
    

def gblinear(data_path, model_name, lead_time, seasons, split_ratio:list,
            max_depth=6, n_rounds=30, learning_rate=0.3, stats=[],
            stats_wind=["max"], stats_pres=["min"],
            df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
            save_path="/users/lpoulain/louis/plots/xgboost/"
            ):
    
    dtrain_wind, dval_wind, dtest_wind,\
    dtrain_pres, dval_pres, dtest_pres,\
    stats_wind, stats_pres, nb_tc,\
    truth_wind_test, truth_pres_test = create_dataset(data_path, model_name, lead_time, seasons, split_ratio,
                                            stats=stats, stats_wind=stats_wind, stats_pres=stats_pres,
                                            df_path=df_path)
    
    params, evallist_wind, evallist_pres = create_booster("gblinear", dtrain_wind, dval_wind, dtrain_pres, dval_pres, max_depth, learning_rate)
    
    # train and save models
    
    wind_save = f"xgb_wind_{model_name}_{lead_time}h_{'_'.join(s for s in seasons)}_depth_{max_depth}_epoch_{n_rounds}_lr_{learning_rate}" +\
                        f"_{'_'.join(stat for stat in stats_wind)}.json"
    pres_save = f"xgb_pres_{model_name}_{lead_time}h_{'_'.join(s for s in seasons)}_depth_{max_depth}_epoch_{n_rounds}_lr_{learning_rate}" +\
                        f"_{'_'.join(stat for stat in stats_pres)}.json"
                        
                        
    bst_wind, eval_res_wind = train_save_xgb(params, dtrain_wind, n_rounds, evallist_wind, early_stopping_rounds=10,
                                            save_path=save_path, save_name=wind_save)
    bst_pres, eval_res_pres = train_save_xgb(params, dtrain_pres, n_rounds, evallist_pres, early_stopping_rounds=10,
                                            save_path=save_path, save_name=pres_save)
    
    # predict and plot
    
    ypred_wind = bst_wind.predict(dtest_wind, iteration_range=(0, bst_wind.best_iteration + 1))
    ypred_pres = bst_pres.predict(dtest_pres, iteration_range=(0, bst_pres.best_iteration + 1))
    
    test_loss_wind = np.sqrt(np.mean((ypred_wind - truth_wind_test)**2))
    test_loss_pres = np.sqrt(np.mean((ypred_pres - truth_pres_test)**2))

    
    plot_xgb(bst_wind, bst_pres, eval_res_wind, eval_res_pres, model_name, "gblinear", lead_time, seasons, nb_tc,
                        max_depth, n_rounds, learning_rate, stats, stats_wind, stats_pres, test_loss_wind, test_loss_pres,
                        save_path="/users/lpoulain/louis/plots/xgboost/",
                        )
    


def dart():
    pass

    
    
if __name__ == "__main__":
    
    fct_dict = {"gbtree":gbtree,
                "dart":dart,
                "gblinear":gblinear,
                }
    
    
    parser = ArgumentParser()
    # paths
    parser.add_argument("--data_path", type=str, default="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/")
    parser.add_argument("--df_path", type=str, 
                        default="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv")
    parser.add_argument("--save_path", type=str, default="/users/lpoulain/louis/plots/xgboost/")
    
    # data params
    parser.add_argument("--split_ratio", type=str2list, default=[0.7,0.2,0.1])
    
    # AI models params
    parser.add_argument("--lead_time", type=int, default=6)
    parser.add_argument("--model", type=str, default="graphcast")
    parser.add_argument("--seasons", type=str2list, default=["2000"])
    
    # type of xgb model
    parser.add_argument('-f', '--func', dest='func', choices=fct_dict.keys(), required=True,
                            help="Choose one of the specified function to be run.")
    
    # stats as input of xgb model
    parser.add_argument("--stats", type=str2list, default=[])
    parser.add_argument("--stats_wind", type=str2list, default=["max"])
    parser.add_argument("--stats_pres", type=str2list, default=["min"])
    
    # xgb model params
    parser.add_argument("--model_args", type=str2list, default=["lr","0.01","depth","2","epochs","20"],
                        help="List of arguments for the model, in the form [arg1,val1,arg2,val2, ...]")
    
    args = parser.parse_args()
    print("Input args:\n", args)
    
    # paths
    data_path, df_path, save_path = args.data_path, args.df_path, args.save_path
    
    # AI models params
    lead_time, model, seasons = args.lead_time, args.model, args.seasons
    
    # xgb method and inputs
    fct = fct_dict[args.func]
    stats, stats_wind, stats_pres = args.stats, args.stats_wind, args.stats_pres
    
    # specific xgb model args
    model_args = {"depth":2, "epochs":20, "lr":0.01, "gamma":0.}
    models_args_dtypes = {"depth":int, "epochs":int, "lr":float, "gamma":float}
    
    for i in range(0, len(args.model_args), 2):
        model_args[args.model_args[i]] = models_args_dtypes[args.model_args[i]](args.model_args[i+1])
    
    print("Model args: ", model_args)
    
    # split ratio
    split_ratio = [float(ratio) for ratio in args.split_ratio]
    
    fct(data_path=data_path, model_name=model, lead_time=lead_time, seasons=seasons, split_ratio=split_ratio,
            max_depth=model_args["depth"], n_rounds=model_args["epochs"], learning_rate=model_args["lr"], gamma=model_args["gamma"], stats=stats,
            stats_wind=stats_wind, stats_pres=stats_pres,
            df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
            save_path="/users/lpoulain/louis/plots/xgboost/")