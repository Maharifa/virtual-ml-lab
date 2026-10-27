import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(page_title="Virtual ML Laboratory: HMM", page_icon="🧪", layout="wide")
st.title("🧪 Virtual Machine Learning Laboratory")
st.header("Lab 1: Hidden Markov Models")
st.markdown("""
**Goal:** Infer a sequence of hidden weather states from observed activities using the Viterbi algorithm.

The hidden state is the weather (**Rainy** or **Sunny**). You can observe an activity (**Walk**, **Shop**, or **Clean**), but not the weather itself. An HMM combines transition probabilities (how weather changes) with emission probabilities (how likely an activity is in each weather state).
""")

states = ["Rainy", "Sunny"]
observations = ["Walk", "Shop", "Clean"]
start_probability = np.array([0.6, 0.4])
transition_probability = np.array([[0.7, 0.3], [0.4, 0.6]])
emission_probability = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])

with st.expander("Model probabilities"):
    st.write("Rows represent the current/hidden weather; columns represent the next weather or observed activity.")
    left, right = st.columns(2)
    left.write("**Transition probabilities** (next weather given current weather)")
    left.dataframe({"Weather": states, "Next: Rainy": transition_probability[:, 0], "Next: Sunny": transition_probability[:, 1]}, hide_index=True)
    right.write("**Emission probabilities** (activity given weather)")
    right.dataframe({"Weather": states, **{activity: emission_probability[:, i] for i, activity in enumerate(observations)}}, hide_index=True)
    st.caption("Initial weather probabilities: Rainy 60%, Sunny 40%.")

sequence_length = st.slider("Number of observations", min_value=5, max_value=20, value=10)
seed = st.number_input("Random seed (change it to generate another sequence)", min_value=0, max_value=10000, value=7, step=1)


def generate_sequence(rng, length):
    hidden = [rng.choice(len(states), p=start_probability)]
    seen = [rng.choice(len(observations), p=emission_probability[hidden[0]])]
    for _ in range(1, length):
        hidden.append(rng.choice(len(states), p=transition_probability[hidden[-1]]))
        seen.append(rng.choice(len(observations), p=emission_probability[hidden[-1]]))
    return np.array(hidden), np.array(seen)


def viterbi_decode(seen):
    # Work in log space to avoid multiplying many small probabilities.
    log_start = np.log(start_probability)
    log_transition = np.log(transition_probability)
    log_emission = np.log(emission_probability)
    scores = np.zeros((len(seen), len(states)))
    backpointers = np.zeros((len(seen), len(states)), dtype=int)
    # candidates[t, destination, predecessor] stores the log score of each route.
    candidates = np.full((len(seen), len(states), len(states)), np.nan)
    scores[0] = log_start + log_emission[:, seen[0]]
    for t in range(1, len(seen)):
        by_predecessor = (
            scores[t - 1][:, None]
            + log_transition
            + log_emission[:, seen[t]][None, :]
        )
        candidates[t] = by_predecessor.T
        backpointers[t] = np.argmax(by_predecessor, axis=0)
        scores[t] = np.max(by_predecessor, axis=0)
    path = np.zeros(len(seen), dtype=int)
    path[-1] = np.argmax(scores[-1])
    for t in range(len(seen) - 2, -1, -1):
        path[t] = backpointers[t + 1, path[t + 1]]
    return path, scores, backpointers, candidates


def draw_state_diagram(path, seen, title):
    """Draw the state at each time, with transition and emission probabilities."""
    fig_width = max(10, len(path) * 1.12)
    fig, ax = plt.subplots(figsize=(fig_width, 4.4))
    y_positions = {0: 0.0, 1: 1.35}
    colors = {0: "#f2a541", 1: "#4385d1"}
    xs = np.arange(len(path))

    # Draw arrows first so state nodes sit on top of them.
    for t in range(1, len(path)):
        prev_state, curr_state = int(path[t - 1]), int(path[t])
        x0, y0 = xs[t - 1], y_positions[prev_state]
        x1, y1 = xs[t], y_positions[curr_state]
        ax.annotate(
            "", xy=(x1 - 0.16, y1), xytext=(x0 + 0.16, y0),
            arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 1.8,
                        "connectionstyle": "arc3,rad=0.12" if prev_state == curr_state else "arc3,rad=0"},
            zorder=1,
        )
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mid_x, mid_y + (0.22 if prev_state == curr_state else 0.12),
                f"a={transition_probability[prev_state, curr_state]:.1f}",
                ha="center", va="center", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1})

    for t, state in enumerate(path):
        state = int(state)
        x, y = xs[t], y_positions[state]
        ax.scatter(x, y, s=760, color=colors[state], edgecolor="white", linewidth=2, zorder=3)
        ax.text(x, y, states[state], ha="center", va="center", color="white", weight="bold", fontsize=9, zorder=4)
        emission = emission_probability[state, int(seen[t])]
        ax.text(x, y - 0.48, f"o{t + 1}: {observations[int(seen[t])]}", ha="center", va="top", fontsize=8)
        ax.text(x, y - 0.70, f"b={emission:.1f}", ha="center", va="top", fontsize=8, color="#475569")
        ax.text(x, 1.95, f"t={t + 1}", ha="center", va="bottom", fontsize=8, color="#475569")

    first = int(path[0])
    ax.text(-0.68, y_positions[first] + 0.26, f"start π={start_probability[first]:.1f}",
            ha="left", va="bottom", fontsize=8, color="#334155")
    ax.set_title(title, loc="left", fontsize=13, weight="bold", pad=16)
    ax.set_xlim(-0.8, len(path) - 0.25)
    ax.set_ylim(-0.88, 2.28)
    ax.set_yticks([y_positions[0], y_positions[1]], labels=states)
    ax.set_xticks([])
    ax.set_xlabel("Time moves from left to right. a = transition probability; b = emission probability.")
    ax.grid(axis="y", alpha=0.15)
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def make_calculation_rows(seen, scores, backpointers, candidates):
    rows = []
    for t, observed in enumerate(seen):
        for destination, state_name in enumerate(states):
            if t == 0:
                start_term = np.log(start_probability[destination])
                emission_term = np.log(emission_probability[destination, observed])
                rows.append({
                    "Time": t + 1,
                    "Observed activity": observations[observed],
                    "Destination state": state_name,
                    "From Rainy candidate (log)": "—",
                    "From Sunny candidate (log)": "—",
                    "Chosen predecessor": "Start",
                    "Winning score δ (log)": f"log π + log b = {start_term:.3f} + {emission_term:.3f} = {scores[t, destination]:.3f}",
                })
            else:
                rain_candidate = candidates[t, destination, 0]
                sun_candidate = candidates[t, destination, 1]
                winner = int(backpointers[t, destination])
                rows.append({
                    "Time": t + 1,
                    "Observed activity": observations[observed],
                    "Destination state": state_name,
                    "From Rainy candidate (log)": f"{rain_candidate:.3f}",
                    "From Sunny candidate (log)": f"{sun_candidate:.3f}",
                    "Chosen predecessor": states[winner],
                    "Winning score δ (log)": f"max(candidates) = {scores[t, destination]:.3f}",
                })
    return rows

if st.button("Generate observations and infer weather", type="primary"):
    rng = np.random.default_rng(int(seed))
    actual_hidden, seen = generate_sequence(rng, int(sequence_length))
    inferred, scores, backpointers, candidates = viterbi_decode(seen)
    correct = int(np.sum(actual_hidden == inferred))

    st.subheader("Observed activity sequence")
    st.write(" → ".join(observations[i] for i in seen))
    st.subheader("State-transition diagram: Viterbi's inferred path")
    st.pyplot(draw_state_diagram(inferred, seen, "Most likely hidden state at each time"), use_container_width=True)
    st.caption("Each node is an inferred hidden state. Arrows show transition probabilities aᵢⱼ; below each node, b is the probability of observing that activity from that state.")

    st.subheader("Viterbi calculation, one time step at a time")
    st.latex(r"\delta_t(j)=\max_i\left[\delta_{t-1}(i)\,a_{ij}\right]b_j(o_t)")
    st.write("For each possible destination state, the table compares the candidate path scores arriving from Rainy and Sunny. The larger candidate is kept; its predecessor is saved for the backward trace.")
    st.dataframe(make_calculation_rows(seen, scores, backpointers, candidates), hide_index=True, use_container_width=True)
    st.caption("The table reports log scores. For example, log(δ × a × b) = log δ + log a + log b. This keeps very small path probabilities numerically stable.")

    with st.expander("Reveal the generated hidden weather (answer key)"):
        st.write("Generated states: " + " → ".join(states[i] for i in actual_hidden))
        st.pyplot(draw_state_diagram(actual_hidden, seen, "Generated hidden states (answer key)"), use_container_width=True)
        st.metric("Positions matching the inferred path", f"{correct} / {len(seen)}")
        st.caption("Viterbi finds the most likely whole state sequence for the observations. It can differ from the random hidden sequence that generated them.")

    st.metric("Log probability of the most likely complete path", f"{np.max(scores[-1]):.3f}")
else:
    st.info("Choose a sequence length and seed, then generate observations to view its state diagram and Viterbi calculations.")

with st.expander("Model probabilities"):
    st.write("Rows represent the current/hidden weather; columns represent the next weather or observed activity.")
    left, right = st.columns(2)
    left.write("**Transition probabilities** (next weather given current weather)")
    left.dataframe({"Weather": states, "Next: Rainy": transition_probability[:, 0], "Next: Sunny": transition_probability[:, 1]}, hide_index=True)
    right.write("**Emission probabilities** (activity given weather)")
    right.dataframe({"Weather": states, **{activity: emission_probability[:, i] for i, activity in enumerate(observations)}}, hide_index=True)
    st.caption("Initial weather probabilities: Rainy 60%, Sunny 40%.")

with st.expander("Think about it"):
    st.markdown("- Follow one arrow: how likely is the next state given the current state?\n- For one table row, which predecessor candidate wins?\n- Why does Viterbi consider the whole sequence instead of guessing each day independently?\n- Why might the inferred path differ from the generated hidden weather?")
