from z3 import *
import random
import numpy as np
import matplotlib.pyplot as plt


STEP = 20


# ----------------------------------
# DSL-like component specification
# ----------------------------------

COMPONENTS = {
    "paper_feed": (100, 300),
    "carriage_system": (150, 400),
    "power_supply": (80, 200),
    "sensors": (20, 100),
    "frame": (200, 700),
    "control_panel": (80, 250)
}


# ----------------------------------
# Build symbolic Z3 model
# ----------------------------------

def build_symbolic_model():

    opt = Optimize()

    cost_vars = {}
    k_vars = {}

    total = 0

    for name, (lo, hi) in COMPONENTS.items():

        cost = Int(name)
        k = Int("k_" + name)

        cost_vars[name] = cost
        k_vars[name] = k

        opt.add(cost == lo + STEP * k)
        opt.add(k >= 0, k <= (hi - lo) // STEP)

        total += cost

    return opt, total


# ----------------------------------
# Compute deterministic bounds once
# ----------------------------------

def compute_deterministic_bounds():

    opt, total = build_symbolic_model()

    # minimum
    opt.push()
    opt.minimize(total)
    opt.check()
    min_val = opt.model().eval(total).as_long()
    opt.pop()

    # maximum
    opt.push()
    opt.maximize(total)
    opt.check()
    max_val = opt.model().eval(total).as_long()
    opt.pop()

    return min_val, max_val


# ----------------------------------
# Probabilistic sampling
# ----------------------------------

def sample_probabilistic():

    ph = int(random.normalvariate(500, 50))
    ph = max(380, min(600, ph))

    cb = int(random.normalvariate(400, 80))
    cb = max(200, min(600, cb))

    ink = random.choice([200, 220, 1000])

    return ph, cb, ink


# ----------------------------------
# Fast Monte Carlo
# ----------------------------------

def run_monte_carlo(min_det, max_det, runs=10000):

    min_results = []
    max_results = []

    for i in range(runs):

        if i % 1000 == 0:
            print(".", end="", flush=True)

        ph, cb, ink = sample_probabilistic()

        base = ph + cb + ink

        min_results.append(base + min_det)
        max_results.append(base + max_det)

    print()

    return min_results, max_results


# ----------------------------------
# ECDF computation
# ----------------------------------

def compute_ecdf(data):

    data = np.sort(np.array(data))
    cdf = np.arange(1, len(data) + 1) / len(data)

    return data, cdf


# ----------------------------------
# Plot ECDFs
# ----------------------------------

def plot_ecdfs(min_results, max_results, filename="printer_cost_ecdf.png"):

    x_min, y_min = compute_ecdf(min_results)
    x_max, y_max = compute_ecdf(max_results)

    plt.figure()

    plt.plot(x_min, y_min, label="Min Cost Resolution")
    plt.plot(x_max, y_max, label="Max Cost Resolution")

    # horizontal reference lines
    plt.axhline(y=0.95, linestyle="--", linewidth=1, label="95%")
    plt.axhline(y=0.99, linestyle=":", linewidth=1, label="99%")

    plt.xlabel("Manufacturing Cost (€)")
    plt.ylabel("ECDF")

    plt.title(
        "ECDF of Canon Printer Manufacturing Cost\n"
        "with Probabilistic Print Head and Controller Board"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(filename, dpi=300)

    plt.show()


# ----------------------------------
# Main
# ----------------------------------

def main():

    min_det, max_det = compute_deterministic_bounds()

    print("Deterministic bounds:")
    print("Min deterministic cost:", min_det)
    print("Max deterministic cost:", max_det)

    min_results, max_results = run_monte_carlo(min_det, max_det, 1000000)

    plot_ecdfs(min_results, max_results)


if __name__ == "__main__":
    main()