import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from SoccerGraph import Player, Graph
from Colony import Colony


# ── Formation Setup ──────────────────────────────────────────────────────────

def make_players(num_extra_defenders=0):
    """7 attackers vs 4+ defenders on a 100x100 pitch, goal at (100,50).

    Striker is placed within 25 units of goal so _can_shoot passes; offside line
    is set by the second-to-last defender from goal (CBs at x=88), so attackers
    at x<=80 stay onside.
    """
    attackers = [
        Player(0, 20, 50, "offense"),   # deep midfielder / ball start
        Player(1, 40, 25, "offense"),   # left midfielder
        Player(2, 40, 75, "offense"),   # right midfielder
        Player(3, 60, 35, "offense"),   # left forward
        Player(4, 60, 65, "offense"),   # right forward
        Player(5, 75, 50, "offense"),   # attacking midfielder
        Player(6, 80, 50, "offense"),   # striker (within shoot range)
    ]
    defenders = [
        Player(7, 97, 50, "defense"),   # goalie
        Player(8, 88, 38, "defense"),   # CB left
        Player(9, 88, 62, "defense"),   # CB right
        Player(10, 70, 50, "defense"),  # defensive mid (high press)
    ]
    # extra defenders placed in midfield to add pressure on passing lanes
    extra_positions = [(55, 30), (55, 70), (45, 50), (65, 40), (65, 60)]
    for i in range(num_extra_defenders):
        x, y = extra_positions[i % len(extra_positions)]
        defenders.append(Player(11 + i, x, y, "defense"))
    return attackers + defenders


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


def plot_convergence(histories, labels, title="Convergence", save_as=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for hist, label in zip(histories, labels):
        ax.plot(range(1, len(hist) + 1), hist, label=label, lw=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best xG")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Helper: run one colony and return history ────────────────────────────────

def run_colony(players, num_ants=1, num_iterations=100, alpha=1.0, beta=2.0, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    graph = Graph(players)
    colony = Colony(graph, num_ants=num_ants, num_iterations=num_iterations,
                    alpha=alpha, beta=beta)
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
        [colony_aco.history, colony_rand.history],
        ["ACO (\u03b1=1, \u03b2=2)", "Random (\u03b1=0, \u03b2=0)"],
        title="ACO vs Random \u2014 Convergence",
        save_as="convergence_aco_vs_random.png",
    )
    return colony_aco


def experiment_defender_density():
    """Experiment 2: How does ACO adapt to increasing defender count?"""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Defender Density")
    print("=" * 60)

    extras = [0, 1, 2, 3, 4]
    histories, labels = [], []

    for n in extras:
        players = make_players(num_extra_defenders=n)
        colony, path, xg = run_colony(players, seed=42)
        total_def = len(Graph(players).get_defenders())
        plen = len(path) if path else 0
        print(f"  {total_def} defenders \u2192 best xG = {xg:.3f}, path len = {plen}")
        histories.append(colony.history)
        labels.append(f"{total_def} def")

    plot_convergence(histories, labels,
                     title="Defender Density \u2014 Convergence",
                     save_as="convergence_defender_density.png")


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
        aco_histories.append(colony_aco.history)
        rand_histories.append(colony_rand.history)

    print(f"  ACO  \u2014 mean: {np.mean(aco_scores):.3f}  std: {np.std(aco_scores):.3f}")
    print(f"  Rand \u2014 mean: {np.mean(rand_scores):.3f}  std: {np.std(rand_scores):.3f}")

    aco_avg = np.mean(aco_histories, axis=0)
    rand_avg = np.mean(rand_histories, axis=0)

    plot_convergence(
        [aco_avg, rand_avg],
        [f"ACO mean (n={n_runs})", f"Random mean (n={n_runs})"],
        title=f"ACO vs Random \u2014 Average Convergence ({n_runs} runs)",
        save_as="convergence_avg_multi_run.png",
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
        histories.append(colony.history)
        labels.append(label)

    plot_convergence(histories, labels,
                     title="\u03b1/\u03b2 Sensitivity \u2014 Convergence",
                     save_as="convergence_alpha_beta.png")


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
    print("\nPlots saved: pitch_aco.png, convergence_aco_vs_random.png,")
    print("  pheromone_heatmap.png, convergence_defender_density.png,")
    print("  convergence_avg_multi_run.png, bar_aco_vs_random.png,")
    print("  convergence_alpha_beta.png")
