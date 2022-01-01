import sys
import time
import numpy as np
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def readCordFile(fpath):
    df = pd.read_csv(fpath)
    X = list(df['X'])
    Y = list(df['Y'])
    return X, Y

def readNNinfo(fpath):
    N = 0
    score = 0
    order = [] 
    with open(fpath) as f:
        ss = f.readline().split('\t')
        N = int(ss[1])
        f.readline()
        f.readline()
        ss = f.readline().split('\t')
        score = float(ss[1])
        ss = f.readline().split('\t')
        order = [int(s) for s in (ss[1].split(','))]
    return N, score, order

def readSAinfo(fpath):
    N = 0
    score = 0
    order = [] 
    with open(fpath) as f:
        ss = f.readline().split('\t')
        N = int(ss[1])
        f.readline()
        f.readline()
        f.readline()
        f.readline()
        f.readline()
        ss = f.readline().split('\t')
        score = float(ss[1])
        ss = f.readline().split('\t')
        order = [int(s) for s in (ss[1].split(','))]
    return N, score, order

def readSAHistory(fpath):
    tcnts = []
    scores = []
    orders = []
    with open(fpath) as f:
        for s in f:
            ss = s.split(',')
            tcnt = int(ss[0])
            score = float(ss[1])
            order = [int(t) for t in ss[2:]]

            tcnts.append(tcnt)
            scores.append(score)
            orders.append(order)
    
    return np.array(tcnts), np.array(scores), np.array(orders)


def saveTSPPNG(fpath, X, Y, ord, savePng = False, showImage = False):
    plotX = [X[i] for i in ord]
    plotY = [Y[i] for i in ord]
    plotXe = [X[ord[-1]], X[ord[0]]]
    plotYe = [Y[ord[-1]], Y[ord[0]]]

    plt.scatter(plotX, plotY)
    plt.plot(plotX, plotY)
    plt.plot(plotXe, plotYe, color='pink')

    if savePng:
        plt.savefig(fpath)
    if showImage:
        plt.show()




def main():
    X, Y = readCordFile("cordinate.csv")
    print(X)
    print(Y)

    #N, ScoreNN, OrderNN = readNNinfo("info_NN.txt")
    #print(N)
    #print(ScoreNN)
    #print(OrderNN)
    #saveTSPPNG("NN.png", X, Y, OrderNN, True, True)

    N, ScoreSA, OrderSA = readSAinfo("info.txt")
    print(N)
    print(ScoreSA)
    print(OrderSA)
    #saveTSPPNG("SA.png", X, Y, OrderSA, True, True)

    tcnts, scores, orders = readSAHistory("orderHistory.csv")
    print(tcnts)
    print(scores)
    print(orders)

    updateindex = []
    mi = 1e18
    for i in range(len(scores)):
        if scores[i] < mi:
            mi = scores[i]
            updateindex.append(i)
    
    print(len(updateindex))
    
    ## animation size params
    #figw, figh, fsize, dsize0, dsize1 = 12.0/3, 6./3., 5, 6, 12
    #figw, figh, fsize, dsize0, dsize1 = 12.0/2, 6./2., 8, 10, 20
    figw, figh, fsize, dsize0, dsize1 = 12.0, 6., 12, 12, 20

    fig, axes = plt.subplots(nrows=1,ncols=2, figsize=(figw,figh))
    plt.rcParams["font.size"] = fsize
    fig.suptitle("TSP: N={0} / SA(2-opt)".format(N))
    def update(idx):
        trgt = updateindex[min(idx, len(updateindex) - 1)]
        axes[0].cla()
        axes[1].cla()
        plotX = [X[i] for i in orders[trgt]]
        plotY = [Y[i] for i in orders[trgt]]
        plotXe = [X[orders[trgt][-1]], X[orders[trgt][0]]]
        plotYe = [Y[orders[trgt][-1]], Y[orders[trgt][0]]]

        axes[0].scatter(plotX, plotY, s = dsize0)
        axes[0].plot(plotX, plotY)
        axes[0].plot(plotXe, plotYe, color='pink')

        axes[1].plot(tcnts, scores, zorder=0)
        axes[1].scatter([tcnts[trgt]],[scores[trgt]], color='red', s = dsize1,zorder=1)
        axes[1].text(5e6,10000, "score: {0:.3f}".format(scores[trgt]))
        axes[1].text(5e6,9500, "update: {0} / {1}".format(min(idx, len(updateindex) - 1) + 1, len(updateindex)))
        axes[1].text(5e6,9000, "accept: {0} / {1}".format(trgt, len(orders)))
        axes[1].text(5e6,8500, "trial: {0} ".format(tcnts[trgt]))
        print("trgt:{0}".format(trgt))

    
    n_holdframe = 30
    ani = animation.FuncAnimation(fig, update, frames=range(len(updateindex) + n_holdframe), interval=80, repeat=False)
    #plt.show()
    w = animation.PillowWriter(fps=12)
    ani.save('animation_TSP_SA_2opt_12fps.gif', writer=w)
    #w = animation.HTMLWriter(fps=12)
    #ani.save('animation_TSP_SA_2opt_12fps.html', writer=w)
    #w = animation.FFMpegWriter(fps=12)
    #ani.save('animation_TSP_SA_2opt_12fps.mp4', writer=w)

    return

if __name__ == "__main__":
    t0 = time.time()
    main()
    print("{0} [sec]".format(time.time() - t0))

