import random

class Latent:
    mu = 0
    sigma = 0.1

    def factor() -> float:
        return random.gauss(Latent.mu, Latent.sigma)

    def factors(k: int = 1) -> list[float]:
        return [Latent.factor() for _ in range(k)]

class User:
    def __init__(self, k: int = 1):
        self.factors = Latent.factors(k)
        self.items = set()

    def add_item(self, item: any):
        self.items.add(item)

class Item:
    def __init__(self, k: int = 1):
        self.factors = Latent.factors(k)
        self.bias = Latent.factor()

def dot(xs: list[float], ys: list[float]) -> float:
    return sum(x * y for x, y in zip(xs, ys))

def recommend(user: Latent, items: dict[any, Latent], top_n=5):
    scores = []
    for item_key, item in items.items():
        if item_key not in user.items:
            scores.append((item_key, dot(user.factors, item.factors)))

    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:top_n]
