# coding: utf-8
import sys
import time
import numpy as np
from collections import deque
import scipy.sparse as sps
import scipy.sparse.linalg as lalg

class DCsolver:

    def __init__(self, nodenum):
        # ソルバ―
        self.linear_solver = lalg.spsolve
        #self.linear_solver = lalg.bicgstab
        #self.linear_solver = lalg.bicg

        # ノード数は固定する．抵抗数とバイアス接続数は更新する
        self.n_node = nodenum
        self.n_res = 0
        self.n_bias_probe = 0

        # 抵抗： 始点，終点，抵抗値
        self.res_from = []
        self.res_to = []
        self.res_value = []

        # 電源： virtual connectするため，名前でアクセスするようにする
        self.bias_name = []
        self.bias_level = []
        self.bias_index = dict() # dict<string,int>

        # Bias supplied (-1: not biased)
        self.biased = [-1] * self.n_node
        # 各バイアスがどのノードにprobeしているかを計算時に確定する
        self.bias_from = None
        self.bias_to = None

        # 1次方程式 A*X=V を解く 
        self._A = None
        self._V = None
        self._X = None 

        # result
        self.node_voltage = None
        self.node_current = None
        self.bias_total_current = None
        self.bias_current_per_node = None
        self.res_current = None

    def add_resistance(self, node_from, node_to, res_value):
        # node_from から node_to へ 抵抗値 res_value の抵抗をつなぐ
        # 　電流の向きをこの向きで定義する
        assert res_value > 0 , "inhibit res_value <= 0"
        self.res_from.append(node_from)
        self.res_to.append(node_to)
        self.res_value.append(res_value)
        self.n_res += 1

    def define_bias(self, bias_name, bias_level=0.0):
        if bias_name in self.bias_index:
            idx = self.bias_index[bias_name]
            self.bias_level[idx] = bias_level
        else :
            idx = len(self.bias_name)
            self.bias_index[bias_name] = idx
            self.bias_name.append(bias_name)
            self.bias_level.append(bias_level)
    
    def supply_bias_to_node(self, bias_name, node_to):
        # 1つのノードに複数バイアスがアクセスするのを禁止して，最も新しい設定を反映する
        assert bias_name in self.bias_index, \
            "{0} is not defined, please define before supply".format(bias_name)
        idx = self.bias_index[bias_name]
        # 既にバイアスが供給されている場合，警告する．
        if self.biased[node_to] != -1:
            print("bias on node:{0} is changed: {1} --> {2} ".format(
                node_to, self.bias_name[self.bias_index[self.biased[node_to]]], bias_name
            ))
            self.biased[node_to] = idx
        else :
            self.biased[node_to] = idx
            self.n_bias_probe += 1

    def _create_matrix(self):
        # (A, V) を定義する
        nv = self.n_node
        nr = self.n_res
        nb = self.n_bias_probe

        # 最終的なバイアス条件設定をリストアップしてインデックス付け
        self.bias_from = []
        self.bias_to = []
        for i in range(nv):
            if self.biased[i] != -1:
                self.bias_from.append(self.biased[i])
                self.bias_to.append(i)
        assert nb == len(self.bias_from)

        # 行列サイズ = ノード数 + 抵抗数 + バイアス供給パス数
        # 未知変数
        #  [0...nv-1]: ノードiの電位
        #  [nv...nv+nr-1] : 抵抗の電流
        #  [nv+nr...n-1] : バイアス供給パスの電流
        n = nv + nr + nb
        mat = sps.lil_matrix((n, n), dtype=np.float64)
        vec = np.zeros(n)

        # Kirchhoff's Current Law （各ノードの outgo と income の総和は0）
        #  i 行目([0...nv-1])の式はノードiに関する電流の和
        #  抵抗jに流れる電流は from[j]のoutgo　to[j]のincomeとしてカウントされる
        for j in range(nr):
            mat[self.res_from[j], nv + j] = 1
            mat[self.res_to[j], nv + j] = -1

        # Kirchhoff's Voltage Law （各抵抗の電圧降下は電位差）
        #  nv+j 行目([nv...nv+nr-1])の式は抵抗j に関する電流の和
        for j in range(nr):
            mat[nv + j, self.res_from[j]] = 1
            mat[nv + j, self.res_to[j]] = -1
            mat[nv + j, nv + j] = -self.res_value[j]
        
        # バイアス定義の式
        #  bias_from[i] の電位がbias_level['bias']に固定される．（KVLパートに追加）
        #  bias_to[i]に電源からの電流が流入する (KCLパートに追加)
        for j in range(len(self.bias_from)):
            mat[nv + nr + j, self.bias_to[j]] = 1
            vec[nv + nr + j] = self.bias_level[self.bias_from[j]]
            mat[self.bias_to[j], nv + nr + j] = -1
        
        # Biasがつながっていないノードがないかチェックする（floating nodeは非許容にする）
        self.check_connention()
        mat = mat.tocsr()
        return mat, vec

    def check_connention(self):
        E = [[] for i in range(self.n_node)]
        for i in range(self.n_res):
            E[self.res_from[i]].append(self.res_to[i])
            E[self.res_to[i]].append(self.res_from[i])
        
        q = deque()
        vis = [False] * self.n_node
        for node in self.bias_to:
            q.append(node)
        while(len(q) > 0):
            now = q.popleft()
            vis[now] = True
            for nxt in E[now]:
                if not vis[nxt]:
                    q.append(nxt)
        
        floating_node = []
        for node in range(len(vis)):
            if not vis[node]:
                floating_node.append(node)
        if len(floating_node) != 0:
            print("Some floating node(s) exist:\nnode(s):\n{0}\n--> Aborted".format(floating_node))
            sys.exit(0)

            

    
    def solve(self):
        A, V = self._create_matrix()
        X = self.linear_solver(A, V)
        X = np.array(X)
        #print(X)
        self._A = A
        self._V = V
        self._X = X

        self.node_voltage = X[0:self.n_node]
        self.res_current = X[self.n_node: self.n_node + self.n_res]        
        self.bias_current_per_node = dict()
        self.bias_total_current = dict()

        for bname in self.bias_name:
            self.bias_current_per_node[bname] = []
            self.bias_total_current[bname] = 0.0
        for i in range(self.n_bias_probe):
            bname = self.bias_name[self.bias_from[i]]
            self.bias_current_per_node[bname].append( (self.bias_to[i], X[self.n_node + self.n_res + i]) )
            self.bias_total_current[bname] += X[self.n_node + self.n_res + i]
        
        # ノードの電流はoutgoのみを計算（（Σ|outgo|+Σ|income|)/2 で計算する）
        self.node_current = np.zeros(self.n_node)
        for i in range(self.n_res):
            self.node_current[self.res_from[i]] += np.abs(self.res_current[i])
            self.node_current[self.res_to[i]] += np.abs(self.res_current[i])
        for bname in self.bias_current_per_node:
            for node, cur in self.bias_current_per_node[bname]:
                self.node_current[node] += np.abs(cur)
        self.node_current /= 2.0

    def print_result_summary(self, showCoef = False):
        if showCoef:
            print('A',self._A)
            print('V',self._V)
            print('X',self._X)
        print('node_voltage\n:{0}\n'.format(self.node_voltage))
        print('res_current\n:{0}\n'.format(self.res_current))
        print('node_cur\n:{0}\n'.format(self.node_current))
        print('bias_cur\n:{0}\n'.format(self.bias_total_current))
        print('bias_cur_per_node\n:{0}\n'.format(self.bias_current_per_node))



def check_serial_connection():
    solver = DCsolver(3)
    solver.add_resistance(0,1,1.0)
    solver.add_resistance(1,2,2.0)
    solver.define_bias('vdd', 1.0)
    solver.define_bias('vss', 0.0)
    solver.supply_bias_to_node('vdd', 0)
    solver.supply_bias_to_node('vss', 2)
    solver.solve()
    solver.print_result_summary(showCoef=True)

def check_parallel_connection():
    solver = DCsolver(2)
    solver.add_resistance(0,1,1.0)
    solver.add_resistance(0,1,2.0)
    solver.define_bias('vdd', 1.0)
    solver.define_bias('vss', 0.0)
    solver.supply_bias_to_node('vdd', 0)
    solver.supply_bias_to_node('vss', 1)
    solver.solve()
    solver.print_result_summary()

def check_mesh_connection(hsize, wsize):
    solver = DCsolver(hsize * wsize)
    def enc(y, x):
        return y * wsize + x
    
    for i in range(hsize - 1):
        for j in range(wsize):
            solver.add_resistance(enc(i,j),enc(i+1,j),1.0)
    for i in range(hsize):
        for j in range(wsize-1):
            solver.add_resistance(enc(i,j),enc(i,j+1),1.0)
    
    solver.define_bias('vdd', 1.0)
    solver.define_bias('vss', 0.0)
    solver.supply_bias_to_node('vdd', 0)
    solver.supply_bias_to_node('vss', enc(hsize-1, wsize - 1))
    solver.solve()
    solver.print_result_summary()



def main():
    check_serial_connection()
    #check_parallel_connection()
    #check_mesh_connection(3,4)

if __name__ == "__main__":
    t0 = time.time()
    main()
    print("{0} [sec]".format(time.time() - t0))

