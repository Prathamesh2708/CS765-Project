# ALL UNITS IN BYTES AND MILLISECONDS

import argparse
import numpy as np
from queue import PriorityQueue
import functools

mining_fee = 50
blk_i_time =  100

np.random.seed(0)

class Transaction:
    id_gen = 0
    def __init__(self, sender_id, receiver_id):
        self.id = Transaction.id_gen
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.size = 1000
        self.coins = -1
        Transaction.id_gen+=1
        
    def __str__(self):
        return f"Txn {self.id}: {self.sender_id} pays {self.receiver_id} {self.coins} coins"

class Block:
    id_gen = 0
    def __init__(self, transaction_set : set, prev_block_id, node_state, depth , creatorID ):
        self.id = Block.id_gen
        self.transaction_set = transaction_set # set of Transaction
        self.node_state = node_state
        self.prev_block_id = prev_block_id
        self.depth = depth
        self.creator = creatorID
        self.size = (len(transaction_set)+1)*1000
        Block.id_gen += 1
    
    def __str__(self):
        return f"Block {self.id}: Creator {self.creator} parent: {self.prev_block_id}"

class Node:
    def __init__(self, id):
        self.id = id
        self.adj = []
        self.txns_seen = set()

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
        self.blockchain = {} # a dictionary mapping id:int to block:Block\
        self.current_head = None

    def initialize_blockchain(self, blockchain, head : Block):
        self.blockchain = blockchain
        self.current_head = head
        self.dangling_blocks = {}
        self.trigger_ids = set()
        self.time_arrived = {}
        
    def _walk_blockchain(self, headid):
        head = self.blockchain[headid]
        while head.id in self.blockchain:
            yield head
            if head.prev_block_id != -1:
                head = self.blockchain[head.prev_block_id]
            else :
                break

    def _verify(self, block : Block) -> bool:
        # you can use _walk_blockchain to do the namesake. It is a generator.
        # check the chain for double spends and also check if the block's parent is present at that node, also check depth of new block is correct, check node state of block
        transaction_set = block.transaction_set
        if block.prev_block_id not in self.blockchain or block.depth-1!=self.blockchain[block.prev_block_id].depth:
            return False 
        prev_state = self.blockchain[block.prev_block_id].node_state.copy()
       
        for transaction in transaction_set:
            prev_state[transaction.sender_id]-=transaction.coins
            prev_state[transaction.receiver_id]+=transaction.coins
        prev_state[block.creator]+=mining_fee
        if block.node_state!=prev_state:
            return False
        for old_block in self._walk_blockchain(block.prev_block_id):
            if transaction_set.intersection(old_block.transaction_set):
                return False
        if old_block.id!=0:
            return False
        else:
            return True
            

    def insert_block(self, block : Block) -> bool:
        if block.id not in self.time_arrived:
            self.time_arrived[block.id] = CURRENT_TIME
        if self._verify(block):
            print(f"Node {self.id} accepts block {block.id}")
            self.blockchain[block.id] = block
            if block.depth > self.current_head.depth:
                self.current_head = block
          
            if block.id in self.trigger_ids:
                removal_set = set()
                for orphan in self.dangling_blocks:
                    if orphan.prev_block_id==block.id and self.insert_block(orphan):
                        removal_set.add(orphan.id)
                for done_id in removal_set:
                    self.dangling_blocks.pop(done_id)
                self.trigger_ids.remove(block.id)        
                
            return True
        else:
            self.dangling_blocks[block.id] = block
            self.trigger_ids.add(block.prev_block_id)
            return False

    def __str__(self):
        return f"Node {self.id} : Coins: {self.coins} Slow: {self.slow} Low: {self.low} Adj: {self.adj}"


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
        
        ##Node can only transact how much money node has, where how much is determined by the state of the head of node's blockchain.
        ##
        self.transaction.coins = np.random.randint(0, nodes[self.transaction.sender_id].current_head.node_state[self.transaction.sender_id]+1)
        # nodes[self.transaction.sender_id].coins -= self.transaction.coins
        # nodes[self.transaction.receiver_id].coins += self.transaction.coins
        # print("Generating ", self.transaction)
        
        ##Add the fact that node has seen its own transaction
        
        ##Broadcast the transaction to all my peers right now.
        eventQueue.put(BroadcastTransactionEvent(self.time, self.transaction,self.transaction.sender_id, nodes[self.transaction.sender_id].adj))
        
        ##Schedule a new event when the next transaction will be done by me.
        n = len(nodes)
        delay = np.random.exponential(args.mean_transaction_delay)
        new_receiver_idx = np.random.randint(0,n)
        while new_receiver_idx == self.transaction.sender_id :
            new_receiver_idx = np.random.randint(0,n)
        eventQueue.put(TransactionEvent(self.time+delay, Transaction(self.transaction.sender_id, new_receiver_idx)))


class BroadcastTransactionEvent(Event):
    def __init__(self, time, transaction : Transaction, transmitter, neighbours) -> None:
        super().__init__(time)
        self.transaction = transaction
        self.transmitter = transmitter
        self.neighbours = neighbours

    def trigger(self, eventQueue, nodes, args):
        if self.transaction not in nodes[self.transaction.sender_id].txns_seen:
            nodes[self.transaction.sender_id].txns_seen.add(self.transaction)
            for neighbour in self.neighbours:
                if self.transaction in nodes[neighbour].txns_seen:
                    # seen already, ignore
                    continue
                # print("Transmitting from ", self.transmitter," to ", neighbour," that ", self.transaction)
                
                c = 5e6 if nodes[self.transmitter].slow or nodes[neighbour].slow else 1e8
                total_delay = args.prop_delay[nodes[self.transmitter].id,nodes[neighbour].id] + self.transaction.size/c + np.random.exponential(scale=96e3/c)
                
                nodes[neighbour].txns_seen.add(self.transaction)
                
                neighbours_neighbours = nodes[neighbour].adj.copy()
                neighbours_neighbours.remove(self.transmitter)
                
                eventQueue.put(BroadcastTransactionEvent(self.time+total_delay, self.transaction,neighbour, neighbours_neighbours))
                

class BlockEvent(Event):
    def __init__(self, time, block_par : Block, creator) -> None:
        super().__init__(time)
        self.block_par = block_par
        self.creator = creator

    def trigger(self, eventQueue, nodes, args):
        ##Check if I can actually transmit a new block
        if nodes[self.creator].current_head == self.block_par:
            
            #set of txns_id
            txns_can_add = nodes[self.creator].txns_seen
            
            #remove those already seen
            txns_can_add = txns_can_add - nodes[self.creator].current_head.transaction_set
            for blk in nodes[self.creator]._walk_blockchain(nodes[self.creator].current_head.id):
                txns_can_add = txns_can_add - blk.transaction_set
            
            #all txns in this set are possible to be added.
            val_txns = list(txns_can_add)[:10]
            blk_txns = set()
            node_state = nodes[self.creator].current_head.node_state.copy()
            for txn in val_txns:
                if node_state[txn.sender_id] > txn.coins:
                    node_state[txn.receiver_id]+=txn.coins
                    node_state[txn.sender_id]-=txn.coins
                    blk_txns.add(txn)
            
            node_state[self.creator]+=mining_fee
            blk = Block(blk_txns ,nodes[self.creator].current_head.id , node_state ,  nodes[self.creator].current_head.depth+1, self.creator )
                
            ##Schedule transmiting this block now
            print(blk)
            eventQueue.put(BroadcastBlockEvent(self.time , blk , self.creator , nodes[self.creator].adj))
            # nodes[self.creator].insert_block(blk)
            
        ##Schedule the next time this creator can make a new block.
        
    
        
        nxt_time = self.time + np.random.exponential(blk_i_time / nodes[self.creator].hash_pow)
        eventQueue.put(BlockEvent(nxt_time , nodes[self.creator].current_head , self.creator))
        


class BroadcastBlockEvent(Event):
    def __init__(self, time, block:Block, transmitter, neighbours) -> None:
        super().__init__(time)
        self.block = block
        self.transmitter = transmitter
        self.neighbours = neighbours

    def trigger(self, eventQueue , nodes , args):
        if self.block.id not in nodes[self.transmitter].time_arrived:
            nodes[self.transmitter].insert_block(self.block)
            for neighbour in self.neighbours:
                if self.block.id in nodes[neighbour].blockchain or self.block.id in nodes[neighbour].dangling_blocks :
                    # seen already, ignore
                    continue
                
                print("Transmitting from ", self.transmitter," to ", neighbour," the block ", self.block)
                
                c = 5e6 if nodes[self.transmitter].slow or nodes[neighbour].slow else 1e8
                total_delay = args.prop_delay[nodes[self.transmitter].id,nodes[neighbour].id] + self.block.size/c + np.random.exponential(scale=96e3/c)
                
                
                neighbours_neighbours = nodes[neighbour].adj.copy()
                neighbours_neighbours.remove(self.transmitter)
                
                eventQueue.put(BroadcastBlockEvent(self.time+total_delay, self.block,neighbour, neighbours_neighbours))



def initialize_nodes(z0, z1, n, GenesisBlock):
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
    slow = n*z0 //100
    for i in range(0,n):
        nodes[i] = Node(i, {
            'coins' : 100,
            'slow' : ((slow_cpus[i]/n)<(z0/100)),
            'cpu'  : ((low_cpus[i]/n) < (z1/100)),
            'adj' : nodes[i].adj
        })
        nodes[i].initialize_blockchain({0 : GenesisBlock}, GenesisBlock)
        print(nodes[i])
        if nodes[i].slow:
            nodes[i].hash_pow = 10/(10*(n-slow) + slow)
        else:
            nodes[i].hash_pow = 1/(10*(n-slow) + slow)
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
    
    init_coins = 100
    
    GenesisBlock = Block(set(), -1, [init_coins]*n, 0, -1)
    nodes = initialize_nodes(z0,z1,n,GenesisBlock)
    eventQueue = PriorityQueue()
    first_transaction_times = np.random.exponential(scale=args.mean_transaction_delay, size=n)
    for i in range(n):
        receiver_idx = np.random.randint(0,n)
        while receiver_idx == i:
            receiver_idx = np.random.randint(0,n)
        eventQueue.put(TransactionEvent(first_transaction_times[i],Transaction(i, receiver_idx)))
        
        eventQueue.put(BlockEvent(np.random.exponential(scale = blk_i_time / nodes[i].hash_pow) , nodes[i].current_head , i))
        
    global CURRENT_TIME
    while not eventQueue.empty():
        event = eventQueue.get()
        CURRENT_TIME =  event.time
        event.trigger(eventQueue, nodes, args)