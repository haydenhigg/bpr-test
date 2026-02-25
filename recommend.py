import json
from mf import User, Item, recommend

with open('./users.json') as f:
    users = {}
    for user_key, user_data in json.load(f).items():
        users[user_key] = User(len(user_data['factors']))
        users[user_key].factors = user_data['factors']
        users[user_key].items = user_data['items']

with open('./items.json') as f:
    items = {}
    for item_key, item_data in json.load(f).items():
        items[item_key] = Item(len(item_data['factors']))
        items[item_key].factors = item_data['factors']
        items[item_key].bias = item_data['bias']

# for f in range(len(list(users.values())[0].factors)):
#     print(f, [x[0] for x in sorted([(item_key, item.factors[f]) for item_key, item in items.items()], key=lambda x: x[1], reverse=True)[:3]])

key = 'haydenhigginbotham'
recommendations = recommend(users[key], items, 3)
print(f"Recommendations for user '{key}':", recommendations)
