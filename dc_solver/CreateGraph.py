# coding: utf-8
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import DCsolver
import DCsolver_sparce
import csv

Solver = DCsolver_sparce

def create_meshDC_from_table(dcmap="map.csv", header="out",show=False, defmap=False, vdd=1.0, vss=0.0):
    t0 = time.time()
    bp = []
    with open(dcmap) as f:
        rdr = csv.reader(f)
        for row in rdr:
            bp.append([int(s) for s in row])
    bp = np.array(bp)
    print(bp.shape)
    print(bp)
    h = bp.shape[0]
    w = bp.shape[1]
    if defmap:
        fig, ax = plt.subplots(1,1,figsize =(4,4))
        sns.heatmap(bp, ax=ax, xticklabels=False, yticklabels=False,cbar=False)
        plt.savefig(header + "_defined.png")
        if show: plt.show()
        return
    
    solver = Solver.DCsolver(h * w)
    def enc(y, x):
        return y * w + x
    
    for i in range(h - 1):
        for j in range(w):
            solver.add_resistance(enc(i,j),enc(i+1,j),1.0)
    for i in range(h):
        for j in range(w-1):
            solver.add_resistance(enc(i,j),enc(i,j+1),1.0)
    
    solver.define_bias('vdd', vdd)
    solver.define_bias('vss', vss)
    for i in range(h):
        for j in range(w):
            if bp[i][j] == 1: solver.supply_bias_to_node('vdd', enc(i,j))
            if bp[i][j] == -1: solver.supply_bias_to_node('vss', enc(i,j))
    
    solver.solve()

    V = solver.node_voltage
    C = solver.node_current
    V = V.reshape((h,w))
    C = C.reshape((h,w))

    fig, ax = plt.subplots(1,1,figsize =(5,4))
    sns.heatmap(V, ax=ax, cmap='jet', xticklabels=False, yticklabels=False)
    plt.savefig(header + "_voltage.png")
    if show: plt.show()
    

    fig, ax = plt.subplots(1,1,figsize =(5,4))
    sns.heatmap(C, ax=ax, cmap='jet', xticklabels=False, yticklabels=False, vmax=0.5)
    plt.savefig(header + "_current.png")
    if show: plt.show()
    print("{0}: {1} [sec]".format(header, time.time() - t0))

def main():
    create_meshDC_from_table("sample1.csv", "sample1_sps", show=False, defmap=True)
    create_meshDC_from_table("sample1.csv", "sample1_sps", show=False, defmap=False)
    create_meshDC_from_table("sample2.csv", "sample2_sps", show=False, defmap=True)
    create_meshDC_from_table("sample2.csv", "sample2_sps", show=False, defmap=False)
    create_meshDC_from_table("sample3.csv", "sample3_sps", show=False, defmap=True)
    create_meshDC_from_table("sample3.csv", "sample3_sps", show=False, defmap=False)

if __name__ == "__main__":
    t0 = time.time()
    main()
    print("{0} [sec]".format(time.time() - t0))

