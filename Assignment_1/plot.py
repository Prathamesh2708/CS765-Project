import pickle
import matplotlib.pyplot as plt

res = []

for i in range(2,10):
    with open(f"results/result_z_0_0.00_z_1_{i*10.0:.2f}_i_100.00","rb") as f:
        out = pickle.load(f)
        res.append(out["ratio of slow nodes to total blocks"])

plt.plot([i*10 for i in range(2,10)], res)
plt.xlabel("z1")
plt.ylabel("ratio of slow nodes to total blocks")
plt.savefig("graphs/ratio of slow nodes to total blocks")