import numpy as np

visits = np.array([10,10,10,10])

probs = visits / visits.sum()
act = np.random.choice(range(4), p = probs)
visits[act] += 1
print(act)
for _ in range(100):
    probs = visits / visits.sum()
    act = np.random.choice(range(4), p = probs)
    visits[act] += 1

print(visits)