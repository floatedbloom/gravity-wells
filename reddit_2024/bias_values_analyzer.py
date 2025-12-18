import matplotlib.pyplot as plt
import numpy as np
import json
from create_conf_bias import ALL_SUBREDDITS

for sub in ALL_SUBREDDITS:

    with open(f'data/bias/bias_values_{sub}.json') as f:
        items = json.load(f)

    items.sort(key=lambda item: item['pairs'], reverse=True)
    pairs = np.array([item['pairs'] for item in items])
    author = np.array([item['author'] for item in items])
    print(sub[:10])
    print(pairs[:10])
    print(author[:10])
    
    # pairs = pairs[pairs != 0.0]
    # print(f'percent zeros: {list(pairs).count(0)/len(pairs)}')
    # print(pairs)
    # plt.hist(pairs, bins=500)
    # plt.show()