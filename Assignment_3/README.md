## Requirements
* python3
* ganache-cli
* truffle
* solidity
* web3
* networkx

## Running Instructions
Run the command
```
ganache-cli -l 1000000000
```
Then in another terminal, compile your truffle project and migrate it to the ganache blockchain
```
truffle compile
truffle migrate
```
Copy the contract address hash and paste it in the client.py at the required place. In another terminal, run
```
python3 client.py
```
The code generates a detailed log of the events happening.
Finally, the result (success rate) is stored in the results.pkl file.