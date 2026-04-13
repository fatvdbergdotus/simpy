from z3 import Optimize, Int, sat
import random
import numpy as np
import matplotlib.pyplot as plt
import time

# ----------------------------------
# Constants
# ----------------------------------

STEP = 20
MONTE_CARLO_RUNS = 10000


# ----------------------------------
# Build solver
# ----------------------------------

def build_solver():

    opt = Optimize()

    # probabilistic variables
    print_head = Int('print_head')
    controller_board = Int('controller_board')
    ink_system = Int('ink_system')

    probabilistic_vars = (print_head, controller_board, ink_system)

    # component costs
    paper_feed = Int('paper_feed')
    carriage_system = Int('carriage_system')
    power_supply = Int('power_supply')
    sensors = Int('sensors')
    frame = Int('frame')
    control_panel = Int('control_panel')

    # step variables
    k_paper = Int('k_paper')
    k_carriage = Int('k_carriage')
    k_power = Int('k_power')
    k_sensors = Int('k_sensors')
    k_frame = Int('k_frame')
    k_panel = Int('k_panel')

    # link costs to step variables
    opt.add(paper_feed == 100 + STEP * k_paper)
    opt.add(carriage_system == 150 + STEP * k_carriage)
    opt.add(power_supply == 80 + STEP * k_power)
    opt.add(sensors == 20 + STEP * k_sensors)
    opt.add(frame == 200 + STEP * k_frame)
    opt.add(control_panel == 80 + STEP * k_panel)

    # bounds
    opt.add(k_paper >= 0, k_paper <= (300 - 100) // STEP)
    opt.add(k_carriage >= 0, k_carriage <= (400 - 150) // STEP)
    opt.add(k_power >= 0, k_power <= (200 - 80) // STEP)
    opt.add(k_sensors >= 0, k_sensors <= (100 - 20) // STEP)
    opt.add(k_frame >= 0, k_frame <= (700 - 200) // STEP)
    opt.add(k_panel >= 0, k_panel <= (250 - 80) // STEP)

    total_cost = Int('total_cost')

    opt.add(total_cost ==
            print_head +
            controller_board +
            ink_system +
            paper_feed +
            carriage_system +
            power_supply +
            sensors +
            frame +
            control_panel)

    return opt, probabilistic_vars, total_cost


# ----------------------------------
# Solve
# ----------------------------------

def solve_cost(opt, probabilistic_vars, total_cost, probabilistic_values):

    opt.push()

    for var, val in zip(probabilistic_vars, probabilistic_values):
        opt.add(var == val)

    # MIN
    opt.push()
    opt.minimize(total_cost)

    min_cost = None
    if opt.check() == sat:
        min_cost = opt.model()[total_cost].as_long()

    opt.pop()

    # MAX
    opt.push()
    opt.maximize(total_cost)

    max_cost = None
    if opt.check() == sat:
        max_cost = opt.model()[total_cost].as_long()

    opt.pop()
    opt.pop()

    return min_cost, max_cost


# ----------------------------------
# Monte Carlo
# ----------------------------------

def sample_probabilistic():

    ph = int(random.normalvariate(500, 50))
    ph = max(380, min(600, ph))

    cb = int(random.normalvariate(400, 80))
    cb = max(200, min(600, cb))

    ink = random.choice([200, 220, 1000])

    return (ph, cb, ink)


def run_monte_carlo(runs=10000):

    opt, probabilistic_vars, total_cost = build_solver()

    min_results = []
    max_results = []

    for i in range(runs):

        if i % 1000 == 0:
            print(".", end="", flush=True)

        values = sample_probabilistic()

        min_cost, max_cost = solve_cost(opt, probabilistic_vars, total_cost, values)

        if min_cost is not None:
            min_results.append(min_cost)
        if max_cost is not None:
            max_results.append(max_cost)

    print()
    return min_results, max_results


# ----------------------------------
# ECDF
# ----------------------------------

def compute_ecdf(data):
    data = np.sort(np.array(data))
    cdf = np.arange(1, len(data) + 1) / len(data)
    return data, cdf


# ----------------------------------
# Inverse ECDF (Quantile)
# ----------------------------------

def compute_inverse_ecdf(data):
    data = np.sort(np.array(data))
    probs = np.arange(1, len(data) + 1) / len(data)
    return probs, data


# ----------------------------------
# Plot ECDF
# ----------------------------------

def plot_ecdfs(min_results, max_results):

    x_min, y_min = compute_ecdf(min_results)
    x_max, y_max = compute_ecdf(max_results)

    plt.figure()

    plt.plot(x_min, y_min, label="Min Cost ECDF")
    plt.plot(x_max, y_max, label="Max Cost ECDF")

    plt.axhline(y=0.95, linestyle="--", linewidth=1, label="95%")
    plt.axhline(y=0.99, linestyle=":", linewidth=1, label="99%")

    plt.xlabel("Manufacturing Cost (€)")
    plt.ylabel("ECDF")
    plt.title("ECDF of Printer Manufacturing Cost")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("printer_cost_ecdf.png", dpi=300)
    plt.show()


# ----------------------------------
# Plot PDF
# ----------------------------------

def plot_pdfs(min_results, max_results):

    plt.figure()

    plt.hist(min_results, bins=50, density=True, alpha=0.5, label="Min PDF")
    plt.hist(max_results, bins=50, density=True, alpha=0.5, label="Max PDF")

    plt.xlabel("Manufacturing Cost (€)")
    plt.ylabel("Density")
    plt.title("PDF of Printer Manufacturing Cost")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("printer_cost_pdf.png", dpi=300)
    plt.show()


# ----------------------------------
# Plot Inverse ECDF
# ----------------------------------

def plot_inverse_ecdfs(min_results, max_results):

    p_min, x_min = compute_inverse_ecdf(min_results)
    p_max, x_max = compute_inverse_ecdf(max_results)

    plt.figure()

    plt.plot(p_min, x_min, label="Min Cost Quantile")
    plt.plot(p_max, x_max, label="Max Cost Quantile")

    for p in [0.5, 0.95, 0.99]:
        plt.axvline(x=p, linestyle="--", linewidth=1)

    plt.xlabel("Probability")
    plt.ylabel("Manufacturing Cost (€)")
    plt.title("Inverse ECDF (Quantile Function)")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("printer_cost_inverse_ecdf.png", dpi=300)
    plt.show()


# ----------------------------------
# Main
# ----------------------------------

def main():

    start = time.perf_counter()

    min_results, max_results = run_monte_carlo(MONTE_CARLO_RUNS)

    end = time.perf_counter()
    print(f"\nExecution time: {end - start:.2f} seconds")

    plot_ecdfs(min_results, max_results)
    plot_pdfs(min_results, max_results)
    plot_inverse_ecdfs(min_results, max_results)


if __name__ == "__main__":
    main()