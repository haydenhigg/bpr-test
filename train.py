import csv
import random
import math
from mf import User, Item, dot, recommend
import json

# hyperparameters
K = 20 # latent factors
LR = 0.05 # learning rate
REG = 0.01 # L2 regularization

EPOCHS = 250_000

# get interactions
with open('my_pacesetter_interactions.csv') as f:
    reader = csv.reader(f)
    interactions = [row for row in reader] # [(user, item)...]

users = {}
items = {}

for user, item in interactions:
    if user not in users:
        users[user] = User(K)

    if item not in items:
        items[item] = Item(K)

    users[user].add_item(item)

# train - matrix factorization using SGD
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

for epoch in range(EPOCHS):
    # gradually decrease learning rate
    if epoch % 10 == 0:
        LR *= 0.9999

    # sample user
    user = random.choice(list(users.values()))

    # sample positive item
    pos_item = items[random.choice(list(user.items))]

    # sample negative item by popularity
    neg_item_key = random.choice(interactions)[1]
    while neg_item_key in user.items:
        neg_item_key = random.choice(interactions)[1]
    neg_item = items[neg_item_key]

    # compute gradient multiplier from score difference
    pos_score = dot(user.factors, pos_item.factors)
    neg_score = dot(user.factors, neg_item.factors)
    grad = 1.0 - sigmoid(pos_score - neg_score)

    # update item biases
    pos_item.bias += LR * (grad - REG * pos_item.bias)
    neg_item.bias += LR * (-grad - REG * neg_item.bias)

    # update latent factors
    for f, uf in enumerate(user.factors):
        pif = pos_item.factors[f]
        nif = neg_item.factors[f]

        user.factors[f] += LR * (grad * (pif - nif) - REG * uf)
        pos_item.factors[f] += LR * (grad * uf - REG * pif)
        neg_item.factors[f] += LR * (-grad * uf - REG * nif)

    print("epoch", epoch, "complete", end='\r')

# save matrices
with open('./users.json', 'w') as f:
    json.dump({user_key: {'factors': user.factors, 'items': list(user.items)} for user_key, user in users.items()}, f)

with open('./items.json', 'w') as f:
    json.dump({item_key: {'factors': item.factors, 'bias': item.bias} for item_key, item in items.items()}, f)
