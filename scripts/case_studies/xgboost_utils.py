import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from loading_utils import statistics_loader, stats_list
from utils.main_utils import multiline_label



def create_dataset(data_path, model_name, lead_time, seasons, split_ratio:list,
            stats=[], stats_wind=["max"], stats_pres=["min"],
            df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
            save_path="/users/lpoulain/louis/plots/xgboost/"
            ):
    
    if stats==[''] or stats==[' ']:
        stats = []
    assert round(sum(split_ratio),5)==1.0, f"split_ratio must sum to 1, your sum is {'+'.join(str(ratio) for ratio in split_ratio)} ="\
                                                + f"{round(sum(split_ratio),5)}"
    assert set(stats).union(stats_wind).union(stats_pres).issubset(set(stats_list)), f"stats/stats_wind/stats_pres must be a subset of {stats_list}.\n"\
                    + f"Your stats are {stats}, stats_wind are {stats_wind} and stats_pres are {stats_pres}"
    
    if type(seasons)==str or type(seasons)==int:
        seasons = [seasons]
        
    # sort stats to always have the same order
    stats_wind.extend(stats)
    stats_pres.extend(stats)
    stats_wind = sorted(list(set(stats_wind)), key=lambda x: stats_list.index(x))
    stats_pres = sorted(list(set(stats_pres)), key=lambda x: stats_list.index(x))
    
    stats_wind_list, stats_pres_list, truth_wind_list, truth_pres_list, tc_ids = statistics_loader(data_path, model_name, lead_time, 
                                                                                                    seasons[0], df_path, save_path)
    
    if len(seasons)>1:
        for season in seasons[1:]:
            tmp_wind_stats, tmp_pres_stats, tmp_wnd_truth, tmp_pres_truth, tmp_ids = statistics_loader(data_path, model_name, lead_time,
                                                                                                    season, df_path, save_path)
            stats_wind_list = {key:val+tmp_wind_stats[key] for key, val in stats_wind_list.items()}
            stats_pres_list = {key:val+tmp_pres_stats[key] for key, val in stats_pres_list.items()}
            truth_wind_list.extend(tmp_wnd_truth)
            truth_pres_list.extend(tmp_pres_truth)
            tc_ids.extend(tmp_ids)
            
    # retain only selected statistics
    
    stats_wind_list = {key:val for key, val in stats_wind_list.items() if key in stats_wind}
    stats_pres_list = {key:val for key, val in stats_pres_list.items() if key in stats_pres}
    
    # shuffle data
    
    rng = np.random.default_rng(73)
    nb_tc = len(tc_ids)
    idx_train, idx_val = int(split_ratio[0]*nb_tc), int((sum(split_ratio[:2]))*nb_tc)
    shuffle_idx = rng.permutation(nb_tc)

    tc_ids = [tc_ids[idx] for idx in shuffle_idx]
    tc_ids_train, tc_ids_val, tc_ids_test = tc_ids[:idx_train], tc_ids[idx_train:idx_val], tc_ids[idx_val:]
    
    stats_wind_list, stats_pres_list = {key:[val[idx] for idx in shuffle_idx] for key, val in stats_wind_list.items()},\
                                       {key:[val[idx] for idx in shuffle_idx] for key, val in stats_pres_list.items()}
    truth_wind_list, truth_pres_list = [truth_wind_list[idx] for idx in shuffle_idx], [truth_pres_list[idx] for idx in shuffle_idx]
    
    # datasets for xgb
    
    stats_wind_train_np, stats_wind_val_np, stats_wind_test_np,\
    stats_pres_train_np, stats_pres_val_np, stats_pres_test_np,\
    truth_wind_train, truth_wind_val, truth_wind_test,\
    truth_pres_train, truth_pres_val, truth_pres_test = split_data(idx_train, idx_val, stats_wind_list, stats_pres_list, 
                                                                   truth_wind_list, truth_pres_list, stats_wind, stats_pres)
            
    dtrain_wind = xgb.DMatrix(stats_wind_train_np, label=truth_wind_train, feature_names=stats_wind)
    dval_wind = xgb.DMatrix(stats_wind_val_np, label=truth_wind_val, feature_names=stats_wind)
    dtest_wind = xgb.DMatrix(stats_wind_test_np, label=truth_wind_test, feature_names=stats_wind)
    
    dtrain_pres = xgb.DMatrix(stats_pres_train_np, label=truth_pres_train, feature_names=stats_pres)
    dval_pres = xgb.DMatrix(stats_pres_val_np, label=truth_pres_val, feature_names=stats_pres)
    dtest_pres = xgb.DMatrix(stats_pres_test_np, label=truth_pres_test, feature_names=stats_pres)

    return dtrain_wind, dval_wind, dtest_wind, dtrain_pres, dval_pres, dtest_pres, stats, stats_wind, stats_pres, nb_tc,\
            truth_wind_test, truth_pres_test


def split_data(idx_train, idx_val, stats_wind_list, stats_pres_list, truth_wind_list, truth_pres_list, stats_wind, stats_pres):
    
    stats_wind_train = {key:val[:idx_train] for key, val in stats_wind_list.items()}
    stats_pres_train = {key:val[:idx_train] for key, val in stats_pres_list.items()}

    truth_wind_train = np.concatenate(truth_wind_list[:idx_train])
    truth_pres_train = np.concatenate(truth_pres_list[:idx_train])
    
    
    stats_wind_val = {key:val[idx_train:idx_val] for key, val in stats_wind_list.items()}
    stats_pres_val = {key:val[idx_train:idx_val] for key, val in stats_pres_list.items()}
    truth_wind_val = np.concatenate(truth_wind_list[idx_train:idx_val])
    truth_pres_val = np.concatenate(truth_pres_list[idx_train:idx_val])
    
    stats_wind_test = {key:val[idx_val:] for key, val in stats_wind_list.items()}
    stats_pres_test = {key:val[idx_val:] for key, val in stats_pres_list.items()}
    truth_wind_test = np.concatenate(truth_wind_list[idx_val:])
    truth_pres_test = np.concatenate(truth_pres_list[idx_val:])
    
    # normalize targets wrt train set
    
    truth_wind_train_mean, truth_wind_train_std = np.mean(truth_wind_train), np.std(truth_wind_train)
    truth_pres_train_mean, truth_pres_train_std = np.mean(truth_pres_train), np.std(truth_pres_train)
    
    truth_wind_train = (truth_wind_train - truth_wind_train_mean)/truth_wind_train_std
    truth_wind_val = (truth_wind_val - truth_wind_train_mean)/truth_wind_train_std
    truth_wind_test = (truth_wind_test - truth_wind_train_mean)/truth_wind_train_std
    
    truth_pres_train = (truth_pres_train - truth_pres_train_mean)/truth_pres_train_std
    truth_pres_val = (truth_pres_val - truth_pres_train_mean)/truth_pres_train_std
    truth_pres_test = (truth_pres_test - truth_pres_train_mean)/truth_pres_train_std
    
    # construct datasets in np.array format
    # input data for xgb should be batch x features (ie [[min1, max1, mean1, std1], [min2, max2, mean2, std2], ...])
    # first, we concatenate for each stat, and then each feature (i.e. each stat)
    stats_wind_train = {key:np.concatenate(val).ravel().reshape(-1, 1) for key, val in stats_wind_train.items()}
    stats_wind_val = {key:np.concatenate(val).ravel().reshape(-1, 1) for key, val in stats_wind_val.items()}
    stats_wind_test = {key:np.concatenate(val).ravel().reshape(-1, 1) for key, val in stats_wind_test.items()}
    
    stats_pres_train = {key:np.concatenate(val).ravel().reshape(-1, 1) for key, val in stats_pres_train.items()}
    stats_pres_val = {key:np.concatenate(val).ravel().reshape(-1, 1) for key, val in stats_pres_val.items()}
    stats_pres_test = {key:np.concatenate(val).ravel().reshape(-1, 1) for key, val in stats_pres_test.items()}
    
    stats_wind_train_np = np.concatenate([stats_wind_train[stat] for stat in stats_wind], axis=1)
    stats_wind_val_np = np.concatenate([stats_wind_val[stat] for stat in stats_wind], axis=1)
    stats_wind_test_np = np.concatenate([stats_wind_test[stat] for stat in stats_wind], axis=1)
    
    stats_pres_train_np = np.concatenate([stats_pres_train[stat] for stat in stats_pres], axis=1)
    stats_pres_val_np = np.concatenate([stats_pres_val[stat] for stat in stats_pres], axis=1)
    stats_pres_test_np = np.concatenate([stats_pres_test[stat] for stat in stats_pres], axis=1)
    
    return stats_wind_train_np, stats_wind_val_np, stats_wind_test_np, stats_pres_train_np, stats_pres_val_np, stats_pres_test_np,\
            truth_wind_train, truth_wind_val, truth_wind_test, truth_pres_train, truth_pres_val, truth_pres_test
            

def create_booster(booster_type, dtrain_wind, dval_wind, dtrain_pres, dval_pres, max_depth, learning_rate, gamma):
    
    param = {}
    param['booster'] = booster_type
    param['verbosity'] = 0
    param['objective'] = 'reg:squarederror'
    param['eval_metric'] = 'rmse'
    
    if booster_type=="gbtree":    
        param['max_depth'] = max_depth
        param['eta'] = learning_rate
        param['gamma'] = gamma
        param['subsample'] = 0.8
        
    if booster_type=="gblinear":
        param['lambda'] = 0.1
        param['alpha'] = 0.1
        param["feature_selector"] = "shuffle"
        
        
    evallist_wind = [(dtrain_wind, 'train'), (dval_wind, 'val')]
    evallist_pres = [(dtrain_pres, 'train'), (dval_pres, 'val')]
    
    return param, evallist_wind, evallist_pres


def cos_annealing(epoch, max_epochs, lr_ini):
    return 0.5 * lr_ini * (1 + np.cos((epoch-max_epochs//2)/max_epochs*np.pi))



def train_save_xgb(params, dtrain, num_round, evallist, early_stopping_rounds=10, callbacks=[],
                   save_path="/users/lpoulain/louis/plots/xgboost/",
                   save_name=""):
    
    eval_res = {}
    bst = xgb.train(params, dtrain, num_round, evallist, evals_result=eval_res, early_stopping_rounds=early_stopping_rounds, callbacks=callbacks)
    
    bst.save_model(save_path + "Models/" + save_name)
    
    return bst, eval_res


def plot_xgb(bst_wind, bst_pres, eval_res_wind, eval_res_pres, model_name, xgb_model, lead_time=0, seasons=0, nb_tc=0,
                max_depth=0, n_rounds=0, learning_rate=0., gamma=0., stats=[], stats_wind=[], stats_pres=[], test_loss_wind=0., 
                test_loss_pres=0., truth_wind_test=None, truth_pres_test=None, y_pred_wind=None, y_pred_pres=None,
                save_path="/users/lpoulain/louis/plots/xgboost/",
                ):
    
    if len(stats)>0:
        for s in stats:
            stats_wind.remove(s)
            stats_pres.remove(s)
    
    
    fig, axs = plt.subplot_mosaic([['a)', 'b)'],
                                   ['c)', 'd)'],
                                   ['e)', 'e)'],
                                   ['f)', 'f)'],
                                   ['g)', 'h)']], figsize=(15, 20))
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axs.values()
    xgb.plot_importance(bst_wind, importance_type="gain", show_values=False, ax=ax1, title="Importance plot - Wind")
    xgb.plot_importance(bst_pres, importance_type="gain", show_values=False, ax=ax2, title="Importance plot - Pressure")
    
    ax3.plot(eval_res_wind['train']['rmse'], label="train loss")
    ax3.plot(eval_res_wind['val']['rmse'], label="val loss")
    ax3.set_title("RMSE losses - Wind")
    ax3.legend()
    
    ax4.plot(eval_res_pres['train']['rmse'], label="train loss")
    ax4.plot(eval_res_pres['val']['rmse'], label="val loss")
    ax4.set_title("RMSE losses - Pressure")
    ax4.legend()
    
    xgb.plot_tree(bst_wind, ax=ax5, num_trees=0)
    ax5.set_title("Tree plot - Wind")
    
    xgb.plot_tree(bst_pres, ax=ax6, num_trees=0)
    ax6.set_title("Tree plot - Pressure")
    
    ax7.hist(y_pred_wind, label="pred", alpha=0.5, bins=100)
    ax7.hist(truth_wind_test, label="truth", alpha=0.5, bins=100)
    ax7.set_title("Prediction vs truth - Wind")
    ax7.legend()
    
    ax8.hist(y_pred_pres, label="pred", alpha=0.5, bins=100)
    ax8.hist(truth_pres_test, label="truth", alpha=0.5, bins=100)
    ax8.set_title("Prediction vs truth - Pressure")
    ax8.legend()
    
    st = fig.suptitle(multiline_label(f"{model_name} - {lead_time}h - {', '.join(s for s in seasons)} ({nb_tc} TCs)"\
            + f"Test losses: {test_loss_wind:.5f} (wind), {test_loss_pres:.5f} (pres)", cutting=3))
    st.set_y(0.98)
    fig.subplots_adjust(bottom=0.005, top=0.90, left=0.05, right=0.95)
    fig.tight_layout()
    
    if int(seasons[0]) + len(seasons)-1==int(seasons[-1]):
        seasons = [seasons[0], "to", seasons[-1]]
    fig.savefig(save_path + "Figs/" + f"{xgb_model}_{model_name}_{lead_time}h_{'_'.join(s for s in seasons)}_depth_"+\
            f"{max_depth}_epoch_{n_rounds}_lr_{learning_rate}_g"+\
            (f"_{gamma}_{'_'.join(stat for stat in stats)}" if len(stats)>0 else "")+\
            (f"_w_{'_'.join(stat for stat in stats_wind)}" if len(stats_wind)>0 else "") +\
            (f"_p_{'_'.join(stat for stat in stats_pres)}" if len(stats_pres)>0 else "") +".png", dpi=500)