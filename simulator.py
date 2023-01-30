# ALL UNITS IN BYTES AND MILLISECONDS

import argparse
import numpy as np
from queue import PriorityQueue
import functools

class Node:
    def __init__(self, id):
        self.id = id
        self.adj = []
        self.txns_seen = {}
        
    
    def __init__(self, id, attr):
        self.adj = []
        self.id = id
        if 'coins' in attr:
            self.coins = attr['coins']
        if 'slow' in attr:
            self.slow  = attr['slow']
        if 'cpu' in attr:
            self.low  = attr['cpu']
        if 'adj' in attr:
            self.adj   = attr['adj']
        self.txns_seen = set()
        
    def __str__(self):
        return f"Node {self.id} : Coins: {self.coins} Slow: {self.slow} Low: {self.low} Adj: {self.adj}"

class Transaction:
    id_gen = 0
    def __init__(self, sender_id, receiver_id, coins):
        self.id = Transaction.id_gen
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.coins = coins
        self.size = 1000
        Transaction.id_gen+=1
        
    def __str__(self):
        return f"Txn {self.id}: {self.sender_id} pays {self.receiver_id} {self.coins} coins"


@functools.total_ordering
class Event:
    def __init__(self, time) -> None:
        self.time = time

    def __gt__(self, other):
        return self.time > other.time

    def __eq__(self, other):
        return self.time == other.time

    def trigger(self, eventQueue, nodes, args):
        raise NotImplementedError()

class TransactionEvent(Event):
    def __init__(self, time, transaction : Transaction) -> None:
        super().__init__(time)
        self.transaction = transaction
        
    def trigger(self, eventQueue, nodes, args):
        nodes[self.transaction.sender_id].coins -= self.transaction.coins
        nodes[self.transaction.receiver_id].coins += self.transaction.coins
        print("Generating ", self.transaction)
        eventQueue.put(BroadcastEvent(self.time, self.transaction,self.transaction.sender_id, nodes[self.transaction.sender_id].adj))
        
        n = len(nodes)
        delay = np.random.exponential(args.mean_transaction_delay)
        new_receiver_idx = np.random.randint(0,n)
        while new_receiver_idx == self.transaction.sender_id :
            new_receiver_idx = np.random.randint(0,n)
            
        eventQueue.put(TransactionEvent(self.time+delay, Transaction(self.transaction.sender_id, new_receiver_idx,np.random.randint(0, nodes[self.transaction.sender_id].coins+1))))


class BroadcastEvent(Event):
    def __init__(self, time, transaction : Transaction, transmitter, neighbours) -> None:
        super().__init__(time)
        self.transaction = transaction
        self.transmitter = transmitter
        self.neighbours = neighbours

    def trigger(self, eventQueue, nodes, args):
        for neighbour in self.neighbours:
            if self.transaction.id in nodes[neighbour].txns_seen:
                # seen already, ignore
                return
            print("Transmitting from ", self.transmitter," to ", neighbour," that ", self.transaction)
            
            c = 5e6 if nodes[self.transmitter].slow or nodes[neighbour].slow else 1e8
            total_delay = args.prop_delay[nodes[self.transmitter].id,nodes[neighbour].id] + self.transaction.size/c + np.random.exponential(scale=96e3/c)
            
            nodes[neighbour].txns_seen.add(self.transaction.id)
            
            neighbours_neighbours = nodes[neighbour].adj.copy()
            neighbours_neighbours.remove(self.transmitter)
            
            eventQueue.put(BroadcastEvent(self.time+total_delay, self.transaction,neighbour, neighbours_neighbours))
            
            


def initialize_nodes(z0, z1, n):
    def checkConnected(nodes):
        vis = [False for n in nodes]
        def dfs(cur , par , vis):
            vis[cur] = True
            for x in nodes[cur].adj:
                if x != par and vis[x] == False:
                    dfs(x , cur,vis)
        dfs( 0 , -1 , vis)
        for i in range(0 , len(nodes)):
            if vis[i] == False:
                return False
        return True
    while True:
        nodes = [Node(i, {}) for i in range(0,n)]
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
        nodes[i] = Node(i, {
            'coins' : 100,
            'slow' : ((slow_cpus[i]/n)<(z0/100)),
            'cpu'  : ((low_cpus[i]/n) < (z1/100)),
            'adj' : nodes[i].adj
        })
        print(nodes[i])
    return nodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--z0' , type = int, default= 0 )
    parser.add_argument('--z1' , type = int, default= 0)
    parser.add_argument('--n' , type = int ,  default= 10)
    parser.add_argument('--mean_transaction_delay', type=float, default=1.0)
    args = parser.parse_args()   
    np.random.seed(0) 
    z0 = args.z0
    z1 = args.z1
    n = args.n
    args.prop_delay = np.random.uniform(10, 500, (n,n))

    nodes = initialize_nodes(z0,z1,n)
    eventQueue = PriorityQueue()
    first_transaction_times = np.random.exponential(scale=args.mean_transaction_delay, size=n)
    for i in range(n):
        receiver_idx = np.random.randint(0,n)
        while receiver_idx == i:
            receiver_idx = np.random.randint(0,n)
        eventQueue.put(TransactionEvent(first_transaction_times[i],Transaction(i, receiver_idx,np.random.randint(0, nodes[i].coins+1))))

    while not eventQueue.empty():
        eventQueue.get().trigger(eventQueue, nodes, args)