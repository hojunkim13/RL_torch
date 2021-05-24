import multiprocessing as mp
import numpy as np
import time
p = (0.3, 0.2, 0.1, 0.1, 0.3)
def sample():
    # Just to make it slow!
    # Pretend that this is a complicated distribution!
    time.sleep(0.1)

    # Re-seed the random number generator
    np.random.seed()

    # Return item index
    return np.argmax(np.random.multinomial(1, p))


def _parallel_mc(iter=1000):
    pool = mp.Pool(4)

    future_res = [pool.apply_async(sample) for _ in range(iter)]
    res = [f.get() for f in future_res]

    return res

def parallel_monte_carlo(iter=1000):
    samples = _parallel_mc(iter)

    # Count each item in our samples
    p = Counter(samples)
    # Convert to dict so we could use update() method properly
    p = dict(p)
    # Normalize them to get the distribution
    p.update([(item, prob / float(iter)) for item, prob in p.items()])

    return p