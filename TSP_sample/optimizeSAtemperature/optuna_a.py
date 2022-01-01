from sys import stdin
import sys
import numpy as np
import collections
from functools import cmp_to_key
import heapq
sys.setrecursionlimit(100000)
import optuna
import os
import time
import pandas as pd
##

def f(t_start, t_end):
    res_txt = "output.txt"
    exe = r".\bin\Debug\netcoreapp3.1\TSP.exe"
    cmd = "{0} {1} {2}  1> {3}".format(exe, t_start, t_end, res_txt)
    print(cmd)
    os.system(cmd)
    time.sleep(0.1)

    with open(res_txt) as f:
        s = f.readline()

    ret = float(s)        
    return ret


def objective(trial):
    t_start = trial.suggest_loguniform('t_start', 1e-10, 1e3)
    t_end = trial.suggest_loguniform('t_end', 1e-10, t_start)
    ret = f(t_start, t_end)
    return ret

def main():
    #ret = f(1, 1e-1)
    #print(ret)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=200)
    hist_df = study.trials_dataframe()
    hist_df.to_csv("optimizing_history.csv")



if __name__ == "__main__":
    main()
