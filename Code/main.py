import yaml
import torch
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torch import multiprocessing

from data import Dataset
from models import get_model
from train_nn import do_train
from utils  import combine_results



def main(argv, tmpdir):
    print("argv", len(argv), argv, tmpdir)
    if len(argv) == 2:
        paths2do = find_pars_yaml(argv, 1)
        num_proc = 1
        dev_ls = [0]
    else: 
        paths2do = find_pars_yaml(argv, 2)
        num_proc = int(argv[1])
        if argv[2] != "cpu":
            dev_ls = [int(x) for x in argv[2].split(".")]
        else:
            dev_ls=["cpu"]
    print("#"*50)
    print("will run "+str(len(paths2do))+" experiments:")
    print(paths2do)
    print("#"*50) 
    multiprocessing.set_start_method("forkserver")
    with multiprocessing.Pool(num_proc) as p:
        for count, path in enumerate(paths2do):
            print((count+1), "of", len(paths2do), "doing", path)
            yml = yaml.safe_load(path.open())
            name = path.parts[-1].split(".")[0]
            yml["name"] = [name]
            save_path = Path("../results/"+name+"/")
            if tmpdir == None:
                print("no tmpdir, set tmp_path to", save_path)
                tmp_path = save_path
            else:
                tmp_path = Path(tmpdir+"/"+name+"/")
            tmp_path.mkdir(parents=True, exist_ok=True)
            run1exp(tmp_path, yml, p, dev_ls)
            combine_results(name, save_path, tmp_path)
            yml["alldone"] = True
            yaml.dump(yml, path.open(mode="w"))
            


def find_pars_yaml(argv, s):
    paths = []
    ret = []
    for x in range(s, len(argv)):
        paths.extend(Path("../Experiments/").glob(argv[x]))
    for p in paths:
        yml = yaml.safe_load(p.open())
        if yml.get("alldone"):
            continue
        else:
            ret.append(p)
    return ret

# # takes one experiments, splits it into the cross product of its settings and starts them
# def run1exp(save_path, update_dict, dev):
#     settings_dict = yaml.safe_load(Path("defaults.yml").open())
#     settings_dict.update(update_dict)
#     setting_ls = list(settings_dict.items())
#     lens = [len(x[1]) for x in setting_ls]
#     keys = [x[0] for x in setting_ls]
#     n = np.prod(np.array(lens))
#     for j in tqdm(range(n)):
#         conf = num2conf(j, lens)
#         current_dict = {keys[i]:settings_dict[keys[i]][conf[i]] for i in range(len(conf))}
#         train1setting(current_dict, save_path/("setting_"+str(j)), dev)

# takes one experiments, splits it into the cross product of its settings and starts them
def run1exp(tmp_path, update_dict, p, dev_ls):
    m = update_dict.get("model", ["N2V"])
    if m == ["N2V"] or m == ["feats"]:
        settings_dict = yaml.safe_load(Path("defaults_n2v.yml").open())
    else:
        settings_dict = yaml.safe_load(Path("defaults_nn.yml").open())
    settings_dict.update(update_dict)
    print(settings_dict)
    print("#"*50)
    setting_ls = list(settings_dict.items())
    lens = [len(x[1]) for x in setting_ls]
    keys = [x[0] for x in setting_ls]
    n = np.prod(np.array(lens))
    it = list(zip(range(n), [lens]*n, [keys]*n, [settings_dict]*n, [tmp_path]*n, [dev_ls]*n))
    tmp = list(tqdm(p.imap(f, it), total=len(it)))

# reduce io on home file system by copying dataset once to a temporary location on job node
#def copy_ds(settings, tmp_path):
    
    

#glue method to use imap
def f(args):
    i, lens, keys, settings_dict, tmp_path, dev_ls = args
    conf = num2conf(i, lens)
    current_dict = {keys[j]:settings_dict[keys[j]][conf[j]] for j in range(len(conf))}
    train1setting(current_dict, tmp_path, dev_ls, i)

    
def num2conf(num, lens):
    left = num
    res = [0]*len(lens)
    for ix in range(len(lens)-1, -1, -1):
        res[ix] = left % lens[ix]
        left = int(left/lens[ix])
    return res

def train1setting(settings, tmp_path, dev_ls, iter_name):

    #Dataset does preprocessing, next 8 lines improve restarting speed of canceld/crashed experiments
    if (tmp_path / ("res_"+str(iter_name)+".pkl")).exists():
        return None

    #yaml.dump(settings, (ws_path/"settings.yml").open(mode="w"))

    pr_nr = int(multiprocessing.current_process().name.split("-")[1])
    if dev_ls == ["cpu"]:
        device = "cpu"
    else:
        device = torch.device("cuda:"+str(dev_ls[pr_nr%len(dev_ls)]))
    settings["device"]=device
    
    ds = Dataset(settings)

    df_ls = []
    for rep in range(settings["statrep"]):
        ds.set_rep(rep)
        torch.manual_seed(rep)
        model = get_model(ds.get_train(), ds.get_train_mask(), settings).to(settings["device"])
        eval_d = do_train(model, ds, settings, rep)
        eval_d["statrep"] = rep
        eval_df = pd.DataFrame(data=eval_d)
        df_ls.append(eval_df)
    df = pd.concat(df_ls, ignore_index=True)
    settings.pop("statrep", None)
    settings.pop("max_epochs", None)
    for k in settings:
        if type(settings[k])==dict or type(settings[k])==list:
            df[k]=str(settings[k])
        else:
            df[k]=settings[k]
    p = tmp_path / ("res_"+str(iter_name)+".pkl")
    df.to_pickle(p)

if __name__ == "__main__":
    if sys.argv[1] == "tmpdir":
        argv = sys.argv
        argv.pop(1)
        d = argv.pop(1)
        main(argv, d)
    else:
        main(sys.argv, None)
