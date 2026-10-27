import math
import random


def sine(x):
    """Smooth target: a sine wave along x1, scaled by x2. x3 and x4 are ignored."""
    return math.sin(math.pi * x[0]) * x[1]


def triangle(x):
    """Cornered target: a triangle wave with the SAME peaks, zeros and period as
    sine(), but made of straight lines with sharp corners at the peaks.
    (2/pi) * asin(sin(pi * x1)) turns the smooth sine into straight zig-zags."""
    return (2 / math.pi) * math.asin(math.sin(math.pi * x[0])) * x[1]


TARGETS = {"sine": sine, "triangle": triangle}


def make_dataset(target="sine", n=500, n_test=100, seed=42):
    """Return (train, test), each a list of (inputs, target) pairs of plain floats.
    The inputs are identical for every target (same seed), only the targets differ,
    so runs on different targets are directly comparable."""
    rule = TARGETS[target]
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        x = [rng.uniform(-1, 1) for _ in range(4)]
        data.append((x, rule(x)))
    return data[: n - n_test], data[n - n_test :]


if __name__ == "__main__":
    for name in TARGETS:
        train, test = make_dataset(name)
        print(f"{name}: {len(train)} train, {len(test)} test")
        for x, y in train[:4]:
            print("  ".join(f"{v:6.3f}" for v in x), f"->  {y:6.3f}")
        print()