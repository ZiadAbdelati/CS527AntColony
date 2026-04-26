import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from SoccerGraph import Player, Graph
from Colony import Colony


# ── Formation Setup ──────────────────────────────────────────────────────────

def make_players(num_extra_defenders=0):
    """8 attackers vs 4+ defenders on a 100x100 pitch, goal at (100,50).

    Two viable shooters (P6 and P7) at different angles so ACO has a real choice.
    Offside line sits at the second-to-last defender from goal (CB at x=88).
    """
    attackers = [
        Player(0, 20, 50, "offense"),   # deep midfielder / ball start
        Player(1, 40, 25, "offense"),   # left midfielder
        Player(2, 40, 75, "offense"),   # right midfielder
        Player(3, 60, 35, "offense"),   # left forward
        Player(4, 60, 65, "offense"),   # right forward
        Player(5, 75, 50, "offense"),   # attacking midfielder
        Player(6, 80, 50, "offense"),   # striker A (central)
        Player(7, 82, 62, "offense"),   # striker B (right side, alternate option)
    ]
    defenders = [
        Player(8, 97, 50, "defense"),   # goalie
        Player(9, 88, 38, "defense"),   # CB left
        Player(10, 88, 62, "defense"),  # CB right
        Player(11, 70, 50, "defense"),  # defensive mid (high press) - the movable one
    ]
    extra_positions = [(55, 30), (55, 70), (45, 50), (65, 40), (65, 60)]
    for i in range(num_extra_defenders):
        x, y = extra_positions[i % len(extra_positions)]
        defenders.append(Player(12 + i, x, y, "defense"))
    return attackers + defenders


# ID of the defensive midfielder (movable defender for adaptive defense)
MOVABLE_DEFENDER_ID = 11
# ID of the deep midfielder where ants always start the play
START_PLAYER_ID = 0


# ── Visualization ────────────────────────────────────────────────────────────

def draw_pitch(graph, path, xg, title="Best Passing Sequence", save_as=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#2e8b57")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect("equal")
    ax.add_patch(patches.Rectangle((0, 0), 100, 100, lw=2, ec="white", fc="none"))
    ax.plot([50, 50], [0, 100], "w--", lw=1, alpha=0.5)
    ax.add_patch(patches.Rectangle((100, 40), 3, 20, lw=2, ec="white", fc="white", alpha=0.4))

    for p in graph.get_attackers():
        ax.plot(*p.pos, "o", color="dodgerblue", ms=12, mec="white", mew=1.5, zorder=5)
        ax.annotate(f"P{p.id}", p.pos, textcoords="offset points",
                    xytext=(0, 10), ha="center", color="white", fontweight="bold", fontsize=9)
    for p in graph.get_defenders():
        ax.plot(*p.pos, "o", color="crimson", ms=12, mec="white", mew=1.5, zorder=5)
        ax.annotate(f"P{p.id}", p.pos, textcoords="offset points",
                    xytext=(0, 10), ha="center", color="white", fontweight="bold", fontsize=9)

    if path and len(path) >= 2:
        for i in range(len(path) - 1):
            ax.annotate("", xy=path[i + 1].pos, xytext=path[i].pos,
                        arrowprops=dict(arrowstyle="-|>", color="yellow", lw=2.5), zorder=4)
    if path:
        ax.annotate("", xy=graph.goal_pos, xytext=path[-1].pos,
                    arrowprops=dict(arrowstyle="-|>", color="orange", lw=2, ls="--"), zorder=4)

    ax.set_title(f"{title}  (xG = {xg:.3f})", fontsize=14, color="white")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.patch.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _smooth(y, window):
    """Centered rolling mean that handles edges by averaging available data."""
    if window <= 1:
        return np.asarray(y, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    half = window // 2
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = y[lo:hi].mean()
    return out


def plot_convergence(histories, labels, title="Convergence", save_as=None,
                     ylabel="Best xG", smooth=1):
    fig, ax = plt.subplots(figsize=(8, 5))
    for hist, label in zip(histories, labels):
        if smooth > 1:
            ax.plot(range(1, len(hist) + 1), hist, lw=1, alpha=0.2)
            ax.plot(range(1, len(hist) + 1), _smooth(hist, smooth),
                    label=label, lw=2.5)
        else:
            ax.plot(range(1, len(hist) + 1), hist, label=label, lw=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Helper: run one colony and return history ────────────────────────────────

def run_colony(players, num_ants=10, num_iterations=100, alpha=2.0, beta=1.0, seed=42,
               start_player_id=START_PLAYER_ID, evaporation_rate=0.02):
    random.seed(seed)
    np.random.seed(seed)
    graph = Graph(players)
    colony = Colony(graph, num_ants=num_ants, num_iterations=num_iterations,
                    alpha=alpha, beta=beta, start_player_id=start_player_id,
                    evaporation_rate=evaporation_rate)
    path, xg = colony.run()
    return colony, path, xg


# ── Experiments ──────────────────────────────────────────────────────────────

def experiment_convergence():
    """Experiment 1 + 3: ACO vs random baseline — convergence and pitch diagram."""
    print("=" * 60)
    print("EXPERIMENT 1: ACO vs Random Baseline")
    print("=" * 60)

    colony_aco, path_aco, xg_aco = run_colony(make_players(), seed=42)
    colony_aco.print_best_path()

    colony_rand, _, xg_rand = run_colony(
        make_players(), alpha=0.0, beta=0.0, seed=42)

    print(f"ACO  best xG: {xg_aco:.3f}")
    print(f"Rand best xG: {xg_rand:.3f}")

    draw_pitch(colony_aco.graph, path_aco, xg_aco,
               title="ACO Best Path", save_as="pitch_aco.png")

    plot_convergence(
        [colony_aco.mean_history, colony_rand.mean_history],
        ["ACO (\u03b1=2, \u03b2=1)", "Random (\u03b1=0, \u03b2=0)"],
        title="ACO vs Random \u2014 Mean xG per Iteration",
        save_as="convergence_aco_vs_random.png",
        ylabel="Mean xG (across ants)",
        smooth=10,
    )
    plot_convergence(
        [colony_aco.diversity_history, colony_rand.diversity_history],
        ["ACO (\u03b1=2, \u03b2=1)", "Random (\u03b1=0, \u03b2=0)"],
        title="Path Diversity \u2014 Unique Paths per Iteration",
        save_as="diversity_aco_vs_random.png",
        ylabel="Unique paths (out of 10 ants)",
        smooth=5,
    )
    return colony_aco


def experiment_defender_density(n_runs=10):
    """Experiment 2: How does ACO adapt to increasing defender count?

    For each defender count, run multiple seeds and average the best xG.
    Plotted as a bar chart with error bars - clearer story than overlapping
    flat lines on a convergence plot.
    """
    print("\n" + "=" * 60)
    print(f"EXPERIMENT 2: Defender Density ({n_runs}-run avg)")
    print("=" * 60)

    extras = [0, 1, 2, 3, 4]
    counts, means, stds = [], [], []

    for n in extras:
        scores = []
        for s in range(n_runs):
            players = make_players(num_extra_defenders=n)
            _, _, xg = run_colony(players, seed=200 + s)
            scores.append(xg)
        total_def = len(Graph(make_players(num_extra_defenders=n)).get_defenders())
        m, sd = float(np.mean(scores)), float(np.std(scores))
        counts.append(total_def); means.append(m); stds.append(sd)
        print(f"  {total_def} defenders \u2192 mean best xG = {m:.3f} \u00b1 {sd:.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([str(c) for c in counts], means, yerr=stds, capsize=5,
                  color="dodgerblue", edgecolor="white")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Number of defenders")
    ax.set_ylabel("Best xG (mean over runs)")
    ax.set_title(f"Defender Density vs Best Achievable xG ({n_runs}-run avg)")
    ax.set_ylim(0, max(means) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("bar_defender_density.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def experiment_multi_run(n_runs=20):
    """ACO vs random over multiple seeds — average convergence."""
    print("\n" + "=" * 60)
    print(f"EXPERIMENT 3: ACO vs Random \u2014 {n_runs}-run average")
    print("=" * 60)

    aco_scores, rand_scores = [], []
    aco_histories, rand_histories = [], []

    for i in range(n_runs):
        s = 100 + i
        colony_aco, _, xg_aco = run_colony(make_players(), seed=s)
        colony_rand, _, xg_rand = run_colony(
            make_players(), alpha=0.0, beta=0.0, seed=s)
        aco_scores.append(xg_aco)
        rand_scores.append(xg_rand)
        aco_histories.append(colony_aco.mean_history)
        rand_histories.append(colony_rand.mean_history)

    print(f"  ACO  \u2014 mean: {np.mean(aco_scores):.3f}  std: {np.std(aco_scores):.3f}")
    print(f"  Rand \u2014 mean: {np.mean(rand_scores):.3f}  std: {np.std(rand_scores):.3f}")

    aco_avg = np.mean(aco_histories, axis=0)
    rand_avg = np.mean(rand_histories, axis=0)

    plot_convergence(
        [aco_avg, rand_avg],
        [f"ACO mean (n={n_runs})", f"Random mean (n={n_runs})"],
        title=f"ACO vs Random \u2014 Mean xG per Iteration ({n_runs} runs)",
        save_as="convergence_avg_multi_run.png",
        ylabel="Mean xG (across ants, avg of runs)",
        smooth=10,
    )

    # bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    means = [np.mean(aco_scores), np.mean(rand_scores)]
    stds = [np.std(aco_scores), np.std(rand_scores)]
    ax.bar(["ACO", "Random"], means, yerr=stds, capsize=5,
           color=["dodgerblue", "gray"], edgecolor="white")
    ax.set_ylabel("Mean Best xG")
    ax.set_title(f"ACO vs Random \u2014 {n_runs}-run comparison")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("bar_aco_vs_random.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def experiment_alpha_beta():
    """Experiment 4: Effect of alpha/beta on convergence."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Alpha/Beta Sensitivity")
    print("=" * 60)

    configs = [
        (0.0, 0.0, "Random (\u03b1=0,\u03b2=0)"),
        (1.0, 0.0, "Pheromone only (\u03b1=1,\u03b2=0)"),
        (0.0, 2.0, "Heuristic only (\u03b1=0,\u03b2=2)"),
        (1.0, 2.0, "ACO (\u03b1=1,\u03b2=2)"),
        (2.0, 3.0, "Strong ACO (\u03b1=2,\u03b2=3)"),
    ]
    histories, labels = [], []

    for alpha, beta, label in configs:
        colony, path, xg = run_colony(make_players(), alpha=alpha, beta=beta, seed=42)
        plen = len(path) if path else 0
        print(f"  {label}: best xG = {xg:.3f}, path len = {plen}")
        histories.append(colony.mean_history)
        labels.append(label)

    plot_convergence(histories, labels,
                     title="\u03b1/\u03b2 Sensitivity \u2014 Mean xG per Iteration",
                     save_as="convergence_alpha_beta.png",
                     ylabel="Mean xG (across ants)",
                     smooth=10)




def experiment_aco_tuning(n_runs=5):
    """Experiment 6: How alpha (pheromone weight) shapes convergence.

    Sweeps alpha with beta=1, evap=0.02 fixed. Plots both mean xG and
    path diversity across iterations, averaged over seeds. Higher alpha =
    pheromones dominate = ants concentrate on fewer paths faster.
    """
    print("\n" + "=" * 60)
    print(f"EXPERIMENT 6: ACO Parameter Tuning ({n_runs}-run avg)")
    print("=" * 60)

    configs = [
        (0.0, 0.0, 0.10, "Random (\u03b1=0, \u03b2=0)"),
        (1.0, 1.0, 0.02, "Weak ACO (\u03b1=1)"),
        (2.0, 1.0, 0.02, "Tuned ACO (\u03b1=2)"),
        (4.0, 1.0, 0.02, "Strong ACO (\u03b1=4)"),
    ]

    mean_curves, div_curves, labels = [], [], []
    for alpha, beta, evap, label in configs:
        means, divs = [], []
        for s in range(n_runs):
            colony, _, _ = run_colony(make_players(), alpha=alpha, beta=beta,
                                      evaporation_rate=evap, seed=200 + s)
            means.append(colony.mean_history)
            divs.append(colony.diversity_history)
        m = np.mean(means, axis=0)
        d = np.mean(divs, axis=0)
        mean_curves.append(m)
        div_curves.append(d)
        labels.append(label)
        print(f"  {label}: end mean xG = {m[-1]:.3f}, end diversity = {d[-1]:.1f}")

    plot_convergence(mean_curves, labels,
                     title="ACO Tuning \u2014 Mean xG per Iteration",
                     save_as="tuning_mean_xg.png",
                     ylabel="Mean xG (across ants)",
                     smooth=10)
    plot_convergence(div_curves, labels,
                     title="ACO Tuning \u2014 Path Diversity",
                     save_as="tuning_diversity.png",
                     ylabel="Unique paths (out of 10 ants)",
                     smooth=5)

# ── Adaptive Defense Experiment ──────────────────────────────────────────────

def experiment_adaptive_defense(total_iterations=100, chunk_size=10, step=4.0, seed=42):
    """Experiment 5: Defender moves to choke off the strongest pheromone lane.

    Every `chunk_size` iterations, find the highest-pheromone edge between
    attackers and shift the movable defender (P11) toward its midpoint.
    Compare convergence to a static-defense baseline on the same seed.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Adaptive Defense")
    print("=" * 60)

    # Static baseline
    random.seed(seed); np.random.seed(seed)
    static_graph = Graph(make_players())
    static_colony = Colony(static_graph, num_ants=1, num_iterations=total_iterations,
                           alpha=1.0, beta=2.0, start_player_id=START_PLAYER_ID)
    static_colony.run()

    # Adaptive run: stitch together chunks, moving defender between them
    random.seed(seed); np.random.seed(seed)
    players = make_players()
    graph = Graph(players)
    movable = next(p for p in graph.get_defenders() if p.id == MOVABLE_DEFENDER_ID)
    trajectory = [tuple(movable.pos.copy())]

    history = []
    best_xg = -1.0
    pheromone_state = None

    n_chunks = total_iterations // chunk_size
    for chunk in range(n_chunks):
        colony = Colony(graph, num_ants=1, num_iterations=chunk_size,
                        alpha=1.0, beta=2.0, start_player_id=START_PLAYER_ID)
        if pheromone_state is not None:
            for i in pheromone_state:
                for j in pheromone_state[i]:
                    if i in colony.pheromone and j in colony.pheromone[i]:
                        colony.pheromone[i][j] = pheromone_state[i][j]
        colony.best_score = best_xg
        colony.run()
        history.extend(colony.history)
        best_xg = colony.best_score
        pheromone_state = colony.pheromone

        # Find strongest attacker–attacker edge
        attackers = graph.get_attackers()
        best_edge = None
        best_strength = -1.0
        for a in attackers:
            for b in attackers:
                if a.id < b.id:
                    s = colony.pheromone[a.id][b.id] + colony.pheromone[b.id][a.id]
                    if s > best_strength:
                        best_strength = s
                        best_edge = (a, b)

        if best_edge is not None:
            a, b = best_edge
            midpoint = (a.pos + b.pos) / 2.0
            direction = midpoint - movable.pos
            dist = np.linalg.norm(direction)
            if dist > 1e-6:
                movable.pos = movable.pos + (direction / dist) * min(step, dist)
            trajectory.append(tuple(movable.pos.copy()))

    print(f"  Static  best xG: {static_colony.best_score:.3f}")
    print(f"  Adaptive best xG: {best_xg:.3f}")
    print(f"  Defender moved {len(trajectory)-1} times, "
          f"start={trajectory[0]}, end={trajectory[-1]}")

    plot_convergence(
        [static_colony.history, history],
        ["Static defense", "Adaptive defense"],
        title="Static vs Adaptive Defense — Convergence",
        save_as="convergence_adaptive_defense.png",
    )

    # Plot defender trajectory on pitch
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#2e8b57")
    ax.set_xlim(-5, 105); ax.set_ylim(-5, 105); ax.set_aspect("equal")
    ax.add_patch(patches.Rectangle((0, 0), 100, 100, lw=2, ec="white", fc="none"))
    ax.add_patch(patches.Rectangle((100, 40), 3, 20, lw=2, ec="white", fc="white", alpha=0.4))
    for p in graph.get_attackers():
        ax.plot(*p.pos, "o", color="dodgerblue", ms=12, mec="white", mew=1.5, zorder=5)
        ax.annotate(f"P{p.id}", p.pos, textcoords="offset points",
                    xytext=(0, 10), ha="center", color="white", fontweight="bold", fontsize=9)
    for p in graph.get_defenders():
        if p.id == MOVABLE_DEFENDER_ID:
            continue
        ax.plot(*p.pos, "o", color="crimson", ms=12, mec="white", mew=1.5, zorder=5)
    xs = [t[0] for t in trajectory]; ys = [t[1] for t in trajectory]
    ax.plot(xs, ys, "-", color="orange", lw=2, alpha=0.7, zorder=4)
    ax.plot(xs, ys, "o", color="crimson", ms=8, mec="white", mew=1, zorder=5)
    ax.plot(xs[0], ys[0], "s", color="yellow", ms=14, mec="black", mew=1.5, zorder=6)
    ax.plot(xs[-1], ys[-1], "*", color="white", ms=20, mec="black", mew=1.5, zorder=6)
    ax.set_title("Adaptive Defender Trajectory (P11)", fontsize=14, color="white")
    fig.patch.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig("adaptive_defense_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Pheromone Heatmap ────────────────────────────────────────────────────────

def plot_pheromone_heatmap(colony, save_as=None):
    graph = colony.graph
    attackers = graph.get_attackers()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#2e8b57")
    ax.set_xlim(-5, 105); ax.set_ylim(-5, 105)
    ax.set_aspect("equal")
    ax.add_patch(patches.Rectangle((0, 0), 100, 100, lw=2, ec="white", fc="none"))

    vals = []
    for a in attackers:
        for b in attackers:
            if a.id != b.id:
                vals.append(colony.pheromone[a.id][b.id])
    max_val = max(vals) if vals else 1.0

    for a in attackers:
        for b in attackers:
            if a.id < b.id:
                strength = colony.pheromone[a.id][b.id] + colony.pheromone[b.id][a.id]
                normed = strength / (2 * max_val)
                if normed > 0.05:
                    ax.plot([a.pos[0], b.pos[0]], [a.pos[1], b.pos[1]],
                            color="yellow", lw=normed * 8, alpha=min(normed + 0.2, 1.0))

    for p in attackers:
        ax.plot(*p.pos, "o", color="dodgerblue", ms=12, mec="white", mew=1.5, zorder=5)
        ax.annotate(f"P{p.id}", p.pos, textcoords="offset points",
                    xytext=(0, 10), ha="center", color="white", fontweight="bold", fontsize=9)
    for p in graph.get_defenders():
        ax.plot(*p.pos, "o", color="crimson", ms=12, mec="white", mew=1.5, zorder=5)

    ax.set_title("Pheromone Strength Between Attackers", fontsize=14, color="white")
    fig.patch.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white")
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    colony = experiment_convergence()
    plot_pheromone_heatmap(colony, save_as="pheromone_heatmap.png")
    experiment_defender_density()
    experiment_multi_run()
    experiment_alpha_beta()
    experiment_aco_tuning()
    experiment_adaptive_defense()
    print("\nPlots saved: pitch_aco.png, convergence_aco_vs_random.png,")
    print("  pheromone_heatmap.png, bar_defender_density.png,")
    print("  convergence_avg_multi_run.png, bar_aco_vs_random.png,")
    print("  convergence_alpha_beta.png, convergence_adaptive_defense.png,")
    print("  adaptive_defense_trajectory.png, diversity_aco_vs_random.png,")
    print("  tuning_mean_xg.png, tuning_diversity.png")
