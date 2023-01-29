import argparse
import numpy as np
class Node:
    def __init__(self):
        self.adj = []
    
    def __init__(self , attr):
        self.adj = []
        if 'coins' in attr:
            self.coins = attr['coins']
        if 'slow' in attr:
            self.slow  = attr['slow']
        if 'cpu' in attr:
            self.low  = attr['cpu']
        if 'adj' in attr:
            self.adj   = attr['adj']
        
    def __string__(self):
        return "Coins :"+self.coins+" Slow :"+self.slow + "Low :" +self.low+" Adj:"+self.adj 
        
def dfs(cur , par , vis):
    vis[cur] = True
    for x in nodes[cur].adj:
        if x != par and vis[x] == False:
            dfs(x , cur,vis)
 
 
def checkConnected(nodes):
    vis = [False for n in nodes]
    dfs( 0 , -1 , vis)
    for i in range(0 , len(nodes)):
        if vis[i] == False:
            return False
    return True
        
parser = argparse.ArgumentParser()
parser.add_argument('--z0' , type = int, default= 0 )
parser.add_argument('--z1' , type = int, default= 0)
parser.add_argument('--n' , type = int ,  default= 10)

if __name__ == "__main__":
    args = parser.parse_args()    
    z0 = args.z0
    z1 = args.z1
    n = args.n
    while True:
        nodes = [Node({}) for i in range(0,n)]
        degs = np.random.randint(low = 4 , high = 9 , size = n)
        if np.sum(degs) % 2 == 1 or np.sum(degs) > n*(n-1)/2:
            continue
        for i in range(0,n):
            thresh = 0
            while (len(nodes[i].adj)< degs[i]) and thresh < 1000:
                x = np.random.randint(n)
                if len(nodes[x].adj) == degs[x] or (i in nodes[x].adj) or i == x:
                    thresh += 1
                    continue
                else:
                    nodes[i].adj.append(x)
                    nodes[x].adj.append(i)
                    thresh = 0
                if thresh == 1000:
                    break
        if thresh == 1000 : 
            continue
        if checkConnected(nodes) == False:
            continue
        cont = False
        for i in range(n):
            if len(nodes[i].adj)!= degs[i]:
                cont = True
        if cont:
            continue
        break
    
    slow_cpus = np.random.permutation(n)
    low_cpus = np.random.permutation(n)
    for i in range(0,n):
        nodes[i] = Node({
            'coins' : 100,
            'slow' : ((slow_cpus[i]/n)<(z0/100)),
            'cpu'  : ((low_cpus[i]/n) < (z1/100)),
            'adj' : nodes[i].adj
        })    
        print("Coins :"+str(nodes[i].coins)+" Slow :"+str(nodes[i].slow) + " Low :" +str(nodes[i].low)+" Adj:"+str(nodes[i].adj) )
    
    