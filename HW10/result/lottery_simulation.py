"""
Thai Government Lottery - Monte Carlo simulation of reward per ticket.
Equivalent to Excel's Random Number Generation (Discrete distribution).
"""
import numpy as np
import pandas as pd

# ---- Prize table -------------------------------------------------------
prizes = pd.DataFrame({
    "prize":   ["1st prize", "Next to 1st", "2nd prize", "3rd prize", "4th prize",
                "5th prize", "First 3 digits", "Last 3 digits", "Last 2 digits"],
    "draws":   [1, 1, 1, 1, 1, 1, 2, 2, 1],
    "tickets": [1, 2, 5, 10, 50, 100, 1_000, 1_000, 10_000],
    "reward":  [6_000_000, 100_000, 200_000, 80_000, 40_000,
                20_000, 4_000, 4_000, 2_000],
})
prizes["prob"] = prizes["draws"] * prizes["tickets"] / 10**6

# Add the "no prize" row so probabilities sum to exactly 1
no_prize = pd.DataFrame([{"prize": "No prize", "draws": 0, "tickets": 0,
                          "reward": 0, "prob": 1 - prizes["prob"].sum()}])
table = pd.concat([prizes, no_prize], ignore_index=True)

values = table["reward"].to_numpy()
probs = table["prob"].to_numpy()
assert np.isclose(probs.sum(), 1.0)

# ---- Theoretical values ------------------------------------------------
mean_theory = np.sum(values * probs)
var_theory = np.sum(values**2 * probs) - mean_theory**2
std_theory = np.sqrt(var_theory)

# ---- Simulation --------------------------------------------------------
N = 1_439_938          # number of random numbers
SEED = 123             # same seed as in Excel (results won't match Excel exactly)

rng = np.random.default_rng(SEED)
# Sample the prize *index* (two prizes share the 4,000 reward, so sampling
# indices lets us count them separately), then map index -> reward
idx = rng.choice(len(values), size=N, p=probs)
sim = values[idx]

# ---- Results -----------------------------------------------------------
print(table.to_string(index=False))
print()
print(f"{'':12}{'Theoretical':>16}{'Simulated':>16}")
print(f"{'Mean':12}{mean_theory:>16,.4f}{sim.mean():>16,.4f}")
print(f"{'Variance':12}{var_theory:>16,.2f}{sim.var(ddof=0):>16,.2f}")
print(f"{'Std (pop)':12}{std_theory:>16,.4f}{sim.std(ddof=0):>16,.4f}")
print(f"{'Std (sample)':12}{'':>16}{sim.std(ddof=1):>16,.4f}")
print(f"{'Min':12}{values.min():>16,}{sim.min():>16,}")
print(f"{'Max':12}{values.max():>16,}{sim.max():>16,}")

# How many times each reward was drawn
print("\nFrequency in simulation:")
freq = pd.DataFrame({"prize": table["prize"], "reward": values,
                     "count": np.bincount(idx, minlength=len(values)),
                     "expected_count": (probs * N).round(1)})
print(freq.to_string(index=False))

# Optional: save the generated numbers (like Excel's output column)
# pd.Series(sim, name="reward").to_csv("lottery_simulation.csv", index=False)
