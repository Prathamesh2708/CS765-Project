# ALL UNITS IN BYTES AND MILLISECONDS
# Node 0 is the malicious one, by default

import argparse
import numpy as np
from queue import PriorityQueue
import functools
import pickle as pkl
from dsplot.graph import Graph
from dsplot.tree import *
import copy
mining_fee = 50
txn_size = 1000

# blk_i_time =  600
#All transactions belong to the Transaction class
class Transaction:
    id_gen = 0
    txn_dict = {}
    def __init__(self, sender_id, receiver_id):
        #initialise all relevant parameters of the transaction
        self.id = Transaction.id_gen
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.size = txn_size
        self.coins = -1
        Transaction.txn_dict[Transaction.id_gen] = self
        Transaction.id_gen+=1
        
            
    def __str__(self):
        return f"Txn {self.id}: {self.sender_id} pays {self.receiver_id} {self.coins} coins"



#A block of the block chain. Stores the id of previous block, the depth of this block and the transactions that this block contains
class Block:
    id_gen = 0
    blocks_dict = {}
    def __init__(self, transaction_set : set, prev_block_id, node_state, depth , creatorID ):
        self.id = Block.id_gen
        self.transaction_set = transaction_set # set of Transaction
        self.node_state = node_state
        self.prev_block_id = prev_block_id
        self.depth = depth
        self.creator = creatorID
        self.size = (len(transaction_set)+1)*txn_size
        Block.blocks_dict[Block.id_gen] = self
        Block.id_gen += 1
    
    def __str__(self):
        return f"Block {self.id}: Creator {self.creator} parent: {self.prev_block_id}"

#The node of our P2P network. Stores details like id,adjacency list, all the txns seen, and other specific details 
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
        self.blockchain = [] # a dictionary mapping id:int to block:Block\
        self.current_head = None

    #A general function, in case this were to be extended to handle nodes which may go down/ get up later on
    def initialize_blockchain(self, blockchain, headId ):
        self.blockchain = blockchain
        self.current_head = headId
        self.dangling_blocks = []
        self.trigger_ids = set()
        self.time_arrived = {}
        
    #A generator of all blocks travelling from the last from to the Genesis block    
    def _walk_blockchain(self, headid):
        head = Block.blocks_dict[headid]
        while head.id in self.blockchain:
            yield head
            if head.prev_block_id != -1:
                head = Block.blocks_dict[head.prev_block_id]
            else :
                break

    #Verifies a block for all its transactions. 
    def _verify(self, block : Block) -> bool:
        
        #check if the block's parent is indeed in the blockchain, and that its depth is 1 more than its parent
        if block.prev_block_id not in self.blockchain or block.depth-1!=Block.blocks_dict[block.prev_block_id].depth:
            return False 
        prev_state = Block.blocks_dict[block.prev_block_id].node_state.copy()
       
        #check if the transactions of the block are valid, given the validity of its parent.
        transaction_set = block.transaction_set
        for txn_id in transaction_set:
            transaction = Transaction.txn_dict[txn_id]
            prev_state[transaction.sender_id]-=transaction.coins
            prev_state[transaction.receiver_id]+=transaction.coins
        prev_state[block.creator]+=mining_fee
        if block.node_state!=prev_state:
            return False
        
        #check that the block does not hold transactions which are present in older blocks
        for old_block in self._walk_blockchain(block.prev_block_id):
            if transaction_set.intersection(old_block.transaction_set):
                return False
        #check that the last ancestor of the block is indeed the Genesis block  
        #This is certainly guaranteed if its parent block was in the blockchain and the conditions were true for it  
        if old_block.id!=0:
            return False
        else:
            return True
            

    def insert_block(self, block : Block) -> bool:
        if block.id not in self.time_arrived:
            self.time_arrived[block.id] = CURRENT_TIME
        if self._verify(block):
            # print(f"Node {self.id} accepts block {block.id}")
            self.blockchain.append(block.id)
            if block.depth > self.current_head.depth:
                self.current_head = block
          
            if block.id in self.trigger_ids:
                removal_set = set()
                for orphan in self.dangling_blocks:
                    if Block.blocks_dict[orphan].prev_block_id==block.id and self.insert_block(Block.blocks_dict[orphan]):
                        removal_set.add(Block.blocks_dict[orphan].id)
                self.dangling_blocks = list(set(self.dangling_blocks)-set(removal_set))
                self.trigger_ids.remove(block.id)        
                
            return True
        else:
            self.dangling_blocks.append(block.id)
            self.trigger_ids.add(block.prev_block_id)
            return False

    def __str__(self):
        return f"Node {self.id} : Coins: {self.coins} Slow: {self.slow} Low: {self.low} Adj: {self.adj}"

class AttackerNode(Node):
    def __init__(self, id):
        super().__init__(id)
        self.private_chain = []
        self.num_rel = 0
        self.highest_added_depth = 0
        self.stubborn = False

    def __init__(self, id, attr):
        super().__init__(id, attr)
        self.private_chain = []
        self.num_rel = 0
        self.highest_added_depth = 0
        if 'stubborn' in attr:
            self.stubborn = attr['stubborn']
        else:
            self.stubborn = False

    def _walk_blockchain(self, headid):
        head = Block.blocks_dict[headid]
        while head.id in self.blockchain+self.private_chain:
            yield head
            if head.prev_block_id != -1:
                head = Block.blocks_dict[head.prev_block_id]
            else :
                break

    def _verify(self, block : Block) -> bool:
        
        #check if the block's parent is indeed in the blockchain, and that its depth is 1 more than its parent
        if block.prev_block_id not in self.blockchain or block.depth-1!=Block.blocks_dict[block.prev_block_id].depth:
            return False 
        prev_state = Block.blocks_dict[block.prev_block_id].node_state.copy()
       
        #check if the transactions of the block are valid, given the validity of its parent.
        transaction_set = block.transaction_set
        for txn_id in transaction_set:
            transaction = Transaction.txn_dict[txn_id]
            prev_state[transaction.sender_id]-=transaction.coins
            prev_state[transaction.receiver_id]+=transaction.coins
        prev_state[block.creator]+=mining_fee
        if block.node_state!=prev_state:
            return False
        
        #check that the block does not hold transactions which are present in older blocks
        for old_block in self._walk_blockchain(block.prev_block_id):
            if transaction_set.intersection(old_block.transaction_set):
                return False
        #check that the last ancestor of the block is indeed the Genesis block  
        #This is certainly guaranteed if its parent block was in the blockchain and the conditions were true for it  
        if old_block.id!=0:
            return False
        else:
            return True

    def _verify_private(self, block : Block) -> bool:
        
        #check if the block's parent is indeed in the blockchain, and that its depth is 1 more than its parent
        if block.prev_block_id not in self.blockchain+self.private_chain or block.depth-1!=Block.blocks_dict[block.prev_block_id].depth:
            return False 
        prev_state = Block.blocks_dict[block.prev_block_id].node_state.copy()
       
        #check if the transactions of the block are valid, given the validity of its parent.
        transaction_set = block.transaction_set
        for txn_id in transaction_set:
            transaction = Transaction.txn_dict[txn_id]
            prev_state[transaction.sender_id]-=transaction.coins
            prev_state[transaction.receiver_id]+=transaction.coins
        prev_state[block.creator]+=mining_fee
        if block.node_state!=prev_state:
            return False
        old_block = block
        #check that the block does not hold transactions which are present in older blocks
        for old_block in self._walk_blockchain(block.prev_block_id):
            if transaction_set.intersection(old_block.transaction_set):
                return False
        #check that the last ancestor of the block is indeed the Genesis block  
        #This is certainly guaranteed if its parent block was in the blockchain and the conditions were true for it  
        if old_block.id!=0:
            return False
        else:
            return True

    def insert_private_block(self, block : Block) -> bool:
        if self._verify_private(block):
            # print(f"Node {self.id} accepts block {block.id}")
            self.private_chain.append(block.id)
            if block.depth > self.current_head.depth:
                self.current_head = block
            return True
        else:
            self.dangling_blocks.append(block.id)
            self.trigger_ids.add(block.prev_block_id)
            return False

    def _policy(self, public_depth, private_depth):
        # modify self.num_rel according to the policy
        if public_depth >= self.highest_added_depth:
            self.highest_added_depth = public_depth
            if self.stubborn:
                delta = private_depth-public_depth
                if delta==1:
                    self.num_rel = 1
                else:
                    self.num_rel = 0
            else:
                self.num_rel = 0
            


    def release(self):
        release_blocks = []
        for blk in self.private_chain:
            if Block.blocks_dict[blk].depth <= self.highest_added_depth + self.num_rel:
                release_blocks.append(blk)
        self.private_chain = list(set(self.private_chain)-set(release_blocks))
        self.num_rel = 0
        return release_blocks


    def insert_block(self, block: Block) -> bool:
        if block.id not in self.time_arrived:
            self.time_arrived[block.id] = CURRENT_TIME
        if self._verify(block):
            # print(f"Node {self.id} accepts block {block.id}")
            self.blockchain.append(block.id)
            if block.depth > self.current_head.depth:
                self.current_head = block
                self.private_chain = []
                self.highest_added_depth = self.current_head.depth
            else:
                self._policy(block.depth, self.current_head.depth)
            if block.id in self.trigger_ids:
                removal_set = set()
                for orphan in self.dangling_blocks:
                    print("orphan", orphan, Block.blocks_dict[orphan].prev_block_id, block.id)
                    if Block.blocks_dict[orphan].prev_block_id==block.id and self.insert_block(Block.blocks_dict[orphan]):
                        removal_set.add(Block.blocks_dict[orphan].id)
                self.dangling_blocks = list(set(self.dangling_blocks)-set(removal_set))
                self.trigger_ids.remove(block.id)        
                
            return True
        else:
            self.dangling_blocks.append(block.id)
            self.trigger_ids.add(block.prev_block_id)
            return False




## A general class for all the events. The events will be scheduled with the help of a priority queue, arranged in the time of their scheduling
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


##The Transaction event. This event creates all the new transactions as well as issues scheduling of the transactions to be broadcasted.
class TransactionEvent(Event):
    def __init__(self, time, transaction : Transaction) -> None:
        super().__init__(time)
        self.transaction = transaction
        
    def trigger(self, eventQueue, nodes, args):
        
        ##Node can only transact how much money node has, where how much is determined by the state of the head of node's blockchain.
        self.transaction.coins = np.random.randint(0, nodes[self.transaction.sender_id].current_head.node_state[self.transaction.sender_id]+1)
        
        ##
        # nodes[self.transaction.sender_id].txns_seen[self.transaction.id] = (self.transaction)
        if args.transaction_print:
            transaction_log.write(f"{CURRENT_TIME}:: Generating {self.transaction} \n")
                
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
        if self.transaction.id not in nodes[self.transmitter].txns_seen:
            nodes[self.transmitter].txns_seen.add(self.transaction.id)
            for neighbour in self.neighbours:
                if self.transaction.id in nodes[neighbour].txns_seen:
                    # seen already, ignore
                    continue
                if args.transaction_print:
                    transaction_log.write(f"{CURRENT_TIME}:: Transmitting from {self.transmitter} to {neighbour}, that {self.transaction} \n")

                c = 5e6 if nodes[self.transmitter].slow or nodes[neighbour].slow else 1e8
                
                total_delay = args.prop_delay[nodes[self.transmitter].id,nodes[neighbour].id] + self.transaction.size/c + np.random.exponential(scale=96e3/c)
                
                # nodes[neighbour].txns_seen.(self.transaction)
                
                neighbours_neighbours = nodes[neighbour].adj.copy()
                neighbours_neighbours.remove(self.transmitter)
                
                ev = BroadcastTransactionEvent(self.time+total_delay, self.transaction,neighbour, neighbours_neighbours)
                
                
                eventQueue.put(ev)
                

##Creates the event of creating new blocks by nodes of the network
class BlockEvent(Event):
    def __init__(self, time, block_par : Block, creator) -> None:
        super().__init__(time)
        self.block_par = block_par
        self.creator = creator

    def trigger(self, eventQueue, nodes, args):
        ##Check if this node can actually transmit a new block
        ##This check is done just by checking that the head on which this block was supposed to be mined is same as the head of the longest chain that I own


        if nodes[self.creator].current_head == self.block_par:
            #set of txns_id
            txns_can_add = set(nodes[self.creator].txns_seen)
            
            #remove those transaction already seen inside the ancestors of this block
            for blk in nodes[self.creator]._walk_blockchain(nodes[self.creator].current_head.id):
                txns_can_add = txns_can_add - blk.transaction_set
            #all txns in this set are now possible to be added.
            val_txns = list(txns_can_add)[:1023]
            blk_txns = set()
            node_state = nodes[self.creator].current_head.node_state.copy()
            for txn_id in val_txns:
                txn = Transaction.txn_dict[txn_id]
                ## Creator checks for the ordering of the transaction inside the block
                if node_state[txn.sender_id] > txn.coins:
                    node_state[txn.receiver_id]+=txn.coins
                    node_state[txn.sender_id]-=txn.coins
                    blk_txns.add(txn_id)
            
            ##Add the coinBase transaction
            node_state[self.creator]+=mining_fee
            blk = Block(blk_txns ,nodes[self.creator].current_head.id , node_state ,  nodes[self.creator].current_head.depth+1, self.creator ) 
            if args.block_print:
                block_log.write(f"{CURRENT_TIME}:: {blk}\n")
            eventQueue.put(BroadcastBlockEvent(self.time , blk , self.creator , nodes[self.creator].adj))

        ##Schedule the next time this creator might make a new block.
        nxt_time = self.time + np.random.exponential(args.blk_i_time / nodes[self.creator].hash_pow)
        eventQueue.put(BlockEvent(nxt_time , nodes[self.creator].current_head , self.creator))
        


class BroadcastBlockEvent(Event):
    def __init__(self, time, block:Block, transmitter, neighbours) -> None:
        super().__init__(time)
        self.block = block
        self.transmitter = transmitter
        self.neighbours = neighbours

    def trigger(self, eventQueue , nodes , args):
        
        if isinstance(nodes[self.transmitter], AttackerNode):
            if self.block.id not in nodes[self.transmitter].time_arrived:
                if self.block.creator == self.transmitter:
                    nodes[self.transmitter].insert_private_block(self.block)
                else:
                    nodes[self.transmitter].insert_block(self.block)
                    release_blocks = nodes[self.transmitter].release()
                    release_blocks.sort()
                    for blk in release_blocks:
                        nodes[self.transmitter].insert_block(Block.blocks_dict[blk])
                        for neighbour in self.neighbours:
                            if blk in nodes[neighbour].blockchain or blk in nodes[neighbour].dangling_blocks :
                                continue
                            if args.block_print:
                                block_log.write(f"{CURRENT_TIME}:: Transmitting from {self.transmitter} to {neighbour} the block { blk}\n")
                            
                            c = 5e6 if nodes[self.transmitter].slow or nodes[neighbour].slow else 1e8
                            total_delay = args.prop_delay[nodes[self.transmitter].id,nodes[neighbour].id] + Block.blocks_dict[blk].size/c + np.random.exponential(scale=96e3/c)
                            
                            
                            neighbours_neighbours = nodes[neighbour].adj.copy()
                            neighbours_neighbours.remove(self.transmitter)
                            
                            eventQueue.put(BroadcastBlockEvent(self.time+total_delay, Block.blocks_dict[blk],neighbour, neighbours_neighbours))
        else:
            if self.block.id not in nodes[self.transmitter].time_arrived:
                nodes[self.transmitter].insert_block(self.block)
                for neighbour in self.neighbours:
                    
                    if self.block.id in nodes[neighbour].blockchain or self.block.id in nodes[neighbour].dangling_blocks :
                        # seen already, ignore
                        continue
                    if args.block_print:
                        block_log.write(f"{CURRENT_TIME}:: Transmitting from {self.transmitter} to {neighbour} the block { self.block}\n")
                    
                    c = 5e6 if nodes[self.transmitter].slow or nodes[neighbour].slow else 1e8
                    total_delay = args.prop_delay[nodes[self.transmitter].id,nodes[neighbour].id] + self.block.size/c + np.random.exponential(scale=96e3/c)
                    
                    
                    neighbours_neighbours = nodes[neighbour].adj.copy()
                    neighbours_neighbours.remove(self.transmitter)
                    
                    eventQueue.put(BroadcastBlockEvent(self.time+total_delay, self.block,neighbour, neighbours_neighbours))



def initialize_nodes(z0, z1, n, GenesisBlock, zeta, args,attacker_node = 0):
    assert(attacker_node in range(n))
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
        nodes = [AttackerNode(i, {}) if i==attacker_node else Node(i, {}) for i in range(0,n)]
        degs = np.random.randint(low = 4 , high = 8 , size = n)
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
    
    
    while len(nodes[attacker_node].adj) < min(zeta*n,n-1):
        honest_node = np.random.randint(n)
        if honest_node != attacker_node and honest_node not in nodes[attacker_node].adj:
            nodes[attacker_node].adj.append(honest_node)
            nodes[honest_node].adj.append(attacker_node)

    for i in range(n):
        nodes[i].adj.sort()
    
    slow_cpus = np.random.permutation(n)
    low_cpus = np.random.permutation(n)
    low = n*z1 //100
    print("*"*10 , "INITIALISATION" , "*"*10)
    for i in range(0,n):
        if i != attacker_node:
            nodes[i] = Node(i, {
                'coins' : 100,
                'slow' : ((slow_cpus[i]/n)<(z0/100)),
                'cpu'  : ((low_cpus[i]/n) < (z1/100)),
                'adj' : nodes[i].adj
            })
        else:
            nodes[i] = AttackerNode(i, {
                'coins' : 100,
                'slow' : False,
                'cpu'  : False,
                'adj' : nodes[i].adj,
                'stubborn' : args.stubborn
            })
            
        nodes[i].initialize_blockchain([0], GenesisBlock)
        print(nodes[i])
        if nodes[i].low:
            nodes[i].hash_pow = 1/(10*(n-low) + low)
        else:
            nodes[i].hash_pow = 10/(10*(n-low) + low)
            
    print("*"*10 , "END OF INITILISATION" , "*"*10)
    return nodes

def low_node_blocks(block_chain , nodes):
    low_blocks = 0
    head = nodes[0].current_head
    while head.id in block_chain:
        if nodes[head.creator].low:
            low_blocks+=1
        if head.id != 0:
            head = Block.blocks_dict[head.prev_block_id]
        else:
            break
            
    # slow_blocks = sum(nodes[block_chain[blk_id].creator].slow for blk_id in block_chain)
    return low_blocks      
  
def number_of_branches(edge_dict):
    in_edge_count = {}
    for key in edge_dict:
        for node in edge_dict[key]:
            if node not in in_edge_count:
                in_edge_count[node] = 1
            else:
                in_edge_count[node] += 1
    total = 0
    for key in in_edge_count:
        total += in_edge_count[key]-1
    return total
    
def average_side_chain_length(edge_dict):
    in_edge_dict = {}
    for key in edge_dict:
        for node in edge_dict[key]:
            if node not in in_edge_dict:
                in_edge_dict[node] = [key]
            else:
                in_edge_dict[node].append(key)
    longest_chain_dict = {}
    side_chain_sum = 0
    side_chain_count = 0
    def _longest(node):
        if node in longest_chain_dict:
            return longest_chain_dict[node]
        else:
            if node in in_edge_dict:
                return 1+max([_longest(i) for i in in_edge_dict[node]])
            else:
                return 1
    _ = _longest(0)
    for node in in_edge_dict:
        if len(in_edge_dict[node]) > 1:
            side_chain_count += len(in_edge_dict[node])-1
            side_chain_sum += sum([_longest(i) for i in in_edge_dict[node]])+1-_longest(node)
    return side_chain_sum/side_chain_count if side_chain_count>0 else 0
    
def average_transactions_per_block(block_chain):
    total_txns_in_blocks = 0
    for blk_id in block_chain:
        total_txns_in_blocks+= Block.blocks_dict[blk_id].size / txn_size
    
    return total_txns_in_blocks / len(block_chain)
  
def len_of_chain(block_chain):
    
    return max([Block.blocks_dict[blk_id].depth for blk_id in block_chain])+1
      
        
## neeche jo bhi functions legenge, like average length

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--z0' , type = int, default= 50 )
    parser.add_argument('--z1' , type = int, default= 20)
    parser.add_argument('--n' , type = int ,  default= 100)
    parser.add_argument('--zeta' , type = float ,  default= 0.25)
    parser.add_argument('-m','--mean_transaction_delay', type=float, default=10.0)
    parser.add_argument('-b','--block_print', action='store_true' )
    parser.add_argument('-t','--transaction_print' , action='store_true')
    parser.add_argument('-i','--blk_i_time',type=float,default = 100.0)
    parser.add_argument('-s','--simulate',type=int,default = 100000)
    parser.add_argument('-st','--stubborn',action='store_true')


    args = parser.parse_args()   
    np.random.seed(0) 
    z0 = args.z0
    z1 = args.z1
    n = args.n
    zeta = args.zeta
    if(zeta<0 or zeta>1):
        print("zeta must be between 0 and 1")
        exit()
    args.prop_delay = np.random.uniform(10, 500, (n,n))
    
    init_coins = 100
    transaction_log = open('txns_log.txt','w+')
    block_log = open('block_log.txt','w+')
    
    
    GenesisBlock = Block(set(), -1, [init_coins]*n, 0, -1)
    nodes = initialize_nodes(z0,z1,n,GenesisBlock, zeta, args,attacker_node = 0)
    eventQueue = PriorityQueue()
    first_transaction_times = np.random.exponential(scale=args.mean_transaction_delay, size=n)
    for i in range(n):
        receiver_idx = np.random.randint(0,n)
        while receiver_idx == i:
            receiver_idx = np.random.randint(0,n)
        eventQueue.put(TransactionEvent(first_transaction_times[i],Transaction(i, receiver_idx)))
        
        eventQueue.put(BlockEvent(np.random.exponential(scale = args.blk_i_time / nodes[i].hash_pow) , nodes[i].current_head , i))
        
    global CURRENT_TIME
    CURRENT_TIME = 0
    while not eventQueue.empty() and CURRENT_TIME < args.simulate :
        event = eventQueue.get()
        CURRENT_TIME =  event.time
        event.trigger(eventQueue, nodes, args)
        
    edge_dict = {0:[]}
    for blk_id in nodes[0].blockchain:
        # print(blk_id)
        if blk_id !=0:
            edge_dict[blk_id]= [Block.blocks_dict[blk_id].prev_block_id]

    # print(edge_dict)
    graph = Graph(edge_dict , directed = True)
    graph.plot(orientation= 'RL', shape = 'square', output_path=f"images/graph0_z_0_{args.z0:.2f}_z_1_{args.z1:.2f}_i_{args.blk_i_time:.2f}.png")  

    edge_dict = {0:[]}
    for blk_id in nodes[4].blockchain:
        # print(blk_id)
        if blk_id !=0:
            edge_dict[blk_id]= [Block.blocks_dict[blk_id].prev_block_id]

    # print(edge_dict)
    graph = Graph(edge_dict , directed = True)
    graph.plot(orientation= 'RL', shape = 'square', output_path=f"images/graph1_z_0_{args.z0:.2f}_z_1_{args.z1:.2f}_i_{args.blk_i_time:.2f}.png")  

    final_block_chain = nodes[0].blockchain
    
    res = {
        "total forks": number_of_branches(edge_dict),
        "average branch length" : average_side_chain_length(edge_dict),
        "average transactions per block": average_transactions_per_block(final_block_chain),
        "ratio of low nodes to total blocks": low_node_blocks(final_block_chain , nodes)/len_of_chain(final_block_chain),
        "ratio of longest chain block to total blocks":len_of_chain(final_block_chain)/len(final_block_chain)
    }

    with open(f"results/result_z_0_{args.z0:.2f}_z_1_{args.z1:.2f}_i_{args.blk_i_time:.2f}","wb") as f:
        pkl.dump(res,f)
    
    '''    Parameter sets: 
    default
    [z_0, z_1, i] = [0.25, 0.25, 600]
    Keep s = 15000 for all simulations for consistency
    z1 -> ratio of number of slow-node-blocks to total-blocks
    z0 -> average length of branch
    i -> number of forks
    i -> number of transaction per block
    '''

    ## ratio of number of slow-node-blocks to total-blocks as a function of z_1
    ## average length of branch as a function of z_0 (network delays)
    ## number of forks vs interarrival time
    ## ratio of blocks in longest-chain to total-blocks generated as a function of z0, z1, Tk
    ## number of transaction per block as a ratio of inerarrival-time to transaction-time