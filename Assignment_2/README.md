## Requirements
* python3
* dsplot

## Running Instructions
Run the code using 
```
python3 simulator.py --z0 0 --z1 20 -i 150 --n 50 -s 6000 -m 100 --zeta 0.25
```
where 
* z0 : fraction of slow nodes
* z1 : fraction of low cpu
* i : interarrival time of blocks
* n : number of nodes
* s : total simulation time
* m : interarrival time of transactions
* zeta : fraction of nodes adversary is connected to

The code generates an image of the blockchain (of one of the nodes) in the images directory and outputs various parameters like number of forks in the results directory as a pickle. 

To get verbose output of blocks and transactions generated/transmitted use the ```-b``` and ```-t``` flags respectively