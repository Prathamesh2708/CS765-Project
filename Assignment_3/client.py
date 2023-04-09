import json
from web3 import Web3
import networkx as nx
import numpy as np
import pickle as pkl

#connect to the local ethereum blockchain
provider = Web3.HTTPProvider('http://127.0.0.1:8545')
w3 = Web3(provider)
#check if ethereum is connected
print(w3.is_connected())

#replace the address with your contract address (!very important)
deployed_contract_address = '0x76d45D87b3b2D8b12490C1Ae2c91645a0B07552e'

#path of the contract json file. edit it with your contract json file
compiled_contract_path ="build/contracts/Payment.json"
with open(compiled_contract_path) as file:
    contract_json = json.load(file)
    contract_abi = contract_json['abi']
contract = w3.eth.contract(address = deployed_contract_address, abi = contract_abi)


print(contract.functions)
'''
#Calling a contract function createAcc(uint,uint,uint)
txn_receipt = contract.functions.createAcc(1, 2, 5).transact({'txType':"0x3", 'from':w3.eth.accounts[0], 'gas':2409638})
txn_receipt_json = json.loads(w3.to_json(txn_receipt))
print(txn_receipt_json) # print transaction hash

# print block info that has the transaction)
print(w3.eth.get_transaction(txn_receipt_json)) 

#Call a read only contract function by replacing transact() with call()

'''

#Add your Code here

n = 100
m = 5
p = 0.25
gas_limit = 1000000

power_graph = nx.powerlaw_cluster_graph(n,m,p)

# Check if the graph is connected
while not nx.is_connected(power_graph):
    power_graph = nx.powerlaw_cluster_graph(n,m,p)

adj_list = nx.to_numpy_array(power_graph)

# Print the adjacency list
print(adj_list.shape)

for i in range(1,n+1):
    txn_receipt = contract.functions.registerUser(i,str(i)).transact({'from':w3.eth.accounts[0]})
    txn_receipt_json = json.loads(w3.to_json(txn_receipt))
    print("Creating Node",i,"Hash", txn_receipt_json) # print transaction hash
    
for i in range(1,n+1):
    for j in range(i+1,n+1):
        if adj_list[i-1][j-1]:
            mean = 10
            amount = round(np.random.exponential(mean))
            txn_receipt = contract.functions.createAcc(i,j,amount).transact({'from':w3.eth.accounts[0]})
            txn_receipt_json = json.loads(w3.to_json(txn_receipt))
            print("Creating Account",i,j,"Amount",amount,"Hash", txn_receipt_json) # print transaction hash

num_txns = 1000
success_arr = np.zeros(num_txns)
for txn in range(num_txns):
    try:
        i,j = np.random.choice(range(1,n+1),2,replace=False)
        txn_receipt = contract.functions.sendAmount(int(i),int(j)).transact({'from':w3.eth.accounts[0]})
        txn_receipt_json = json.loads(w3.to_json(txn_receipt))
        print("Send Amount",i,j,"Hash",txn_receipt_json,"Status",contract.events.TxnStatus.get_logs()[0].args['status']) # print transaction hash
        success_arr[txn] = contract.events.TxnStatus.get_logs()[0].args['status']
    except:
        txn-=1
print(success_arr)

with open('results2.pkl','wb') as f:
    pkl.dump(success_arr,f)













