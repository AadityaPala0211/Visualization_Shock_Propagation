from __future__ import annotations
from collections import deque
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import streamlit as st
from scipy import sparse
from scipy.sparse.linalg import eigs

# Place the research repository's results_network_analysis folder beside app.py.
# Expected local paths mirror the research repository's directory structure.
RESULTS_NETWORK_DIR = Path(__file__).resolve().parent / "results_network_analysis"
WD_MATRIX_PATH = RESULTS_NETWORK_DIR / "empirical" / "2022" / "WD" / "adjacency.npz"
NODE_INDEX_PATH = RESULTS_NETWORK_DIR / "metadata" / "node_index.csv"
NODE_METRICS_PATH = RESULTS_NETWORK_DIR / "empirical" / "2022" / "WD" / "node_metrics.csv.gz"
SECTOR_SUMMARY_PATH = RESULTS_NETWORK_DIR / "empirical" / "2022" / "WD" / "sector_summary.csv"

@st.cache_data(show_spinner=False)
def load_network():
    required_files = [WD_MATRIX_PATH, NODE_INDEX_PATH, NODE_METRICS_PATH, SECTOR_SUMMARY_PATH]
    missing_files = [str(path.relative_to(Path(__file__).resolve().parent))
                     for path in required_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "The local 2022 research data files are missing. Expected these paths "
            "relative to the folder containing app.py: " + ", ".join(missing_files)
        )
    matrix = sparse.load_npz(WD_MATRIX_PATH).tocsr()
    index = pd.read_csv(NODE_INDEX_PATH)
    metrics = pd.read_csv(NODE_METRICS_PATH, compression="gzip")
    sectors = pd.read_csv(SECTOR_SUMMARY_PATH)
    if matrix.shape != (4050, 4050) or len(index) != 4050:
        raise ValueError("Expected the aligned 4,050-node 2022 research network.")
    index = index.sort_values("node_id").reset_index(drop=True)
    metrics = metrics.sort_values("node_id").reset_index(drop=True)
    if not np.array_equal(index.node_id.to_numpy(), metrics.node_id.to_numpy()):
        raise ValueError("Node index and metrics have inconsistent ordering.")
    matrix.setdiag(0); matrix.eliminate_zeros()
    return matrix, index, metrics, sectors

@st.cache_data(show_spinner=False)
def get_spectral_radius(_matrix, network_version):
    # Streamlit skips hashing the SciPy CSR matrix; network_version invalidates
    # the cached value whenever the local adjacency file is replaced.
    if _matrix.nnz == 0: return 0.0
    return float(abs(eigs(_matrix, k=1, which="LM", return_eigenvectors=False, tol=1e-5, maxiter=3000)[0]))

@st.cache_data(show_spinner="Calculating shock propagation…")
def run_sis(_B, source, severity, beta, rho, steps, network_version):
    # _B is excluded from hashing because SciPy CSR matrices are not hashable.
    # The fixed local network is represented in the cache key by network_version.
    state = np.zeros(_B.shape[0], dtype=float)
    state[source] = severity
    history = [state.copy()]
    BT = _B.T.tocsr()
    for _ in range(steps):
        exposure = np.asarray(BT @ state).ravel()
        state = np.clip((1-rho)*state + (1-state)*beta*exposure, 0, 1)
        history.append(state.copy())
        if state.max() < 1e-8: break
    return np.asarray(history)

def get_sector(label): return label.split("_", 1)[1] if "_" in label else label
def get_country(label): return label.split("_", 1)[0] if "_" in label else ""

def focus_neighbourhood(B, focus, max_neighbours=28):
    """Return strongest incoming and outgoing neighbours of one focal activity."""
    outgoing = B.getrow(focus)
    incoming = B.getcol(focus).tocoo()
    # Keep both directions visible; split the neighbour budget between them.
    half = max_neighbours // 2
    out_pairs = sorted(zip(outgoing.indices.tolist(), outgoing.data.tolist()),
                       key=lambda item: item[1], reverse=True)[:half]
    in_pairs = sorted(zip(incoming.row.tolist(), incoming.data.tolist()),
                      key=lambda item: item[1], reverse=True)[:max_neighbours-half]
    neighbours = {}
    for node, weight in out_pairs:
        if int(node) != focus:
            neighbours[int(node)] = max(neighbours.get(int(node), 0.0), float(weight))
    for node, weight in in_pairs:
        if int(node) != focus:
            neighbours[int(node)] = max(neighbours.get(int(node), 0.0), float(weight))
    # If one direction has few links, fill remaining slots from the other direction.
    if len(neighbours) < max_neighbours:
        candidates = sorted(
            [(int(n), float(w)) for n, w in zip(outgoing.indices, outgoing.data)] +
            [(int(n), float(w)) for n, w in zip(incoming.row, incoming.data)],
            key=lambda item: item[1], reverse=True)
        for node, weight in candidates:
            if node != focus and node not in neighbours:
                neighbours[node] = weight
                if len(neighbours) >= max_neighbours:
                    break
    return [focus] + list(neighbours.keys()), neighbours

def make_figure(B, labels, state, focus, max_neighbours=28):
    """Fast radial ego view: no force simulation, only links touching the focus."""
    chosen, weights = focus_neighbourhood(B, focus, max_neighbours)
    neighbour_ids = chosen[1:]
    count = len(neighbour_ids)
    positions = {focus: (0.0, 0.0)}
    if count:
        # Deterministic circular layout: O(number of displayed neighbours).
        angles = np.linspace(0, 2*np.pi, count, endpoint=False) + np.pi/2
        for node, angle in zip(neighbour_ids, angles):
            positions[node] = (float(np.cos(angle)), float(np.sin(angle)))

    fig = go.Figure()
    # Draw only edges incident to the focal activity, not every edge in a multi-hop subgraph.
    line_x, line_y = [], []
    for node in neighbour_ids:
        x0, y0 = positions[focus]
        x1, y1 = positions[node]
        line_x.extend([x0, x1, None])
        line_y.extend([y0, y1, None])
    fig.add_trace(go.Scatter(
        x=line_x, y=line_y, mode="lines",
        line=dict(width=1.1, color="rgba(130,145,165,.30)"),
        hoverinfo="skip", name="Connections"
    ))

    # Directional arrows: incoming suppliers point into focus; outgoing buyers point away.
    for node in neighbour_ids:
        node_to_focus = float(B[node, focus])
        focus_to_node = float(B[focus, node])
        x1, y1 = positions[node]
        pressure_in = float(state[node]) * node_to_focus
        pressure_out = float(state[focus]) * focus_to_node
        for is_incoming, weight, pressure in (
            (True, node_to_focus, pressure_in),
            (False, focus_to_node, pressure_out)
        ):
            if weight <= 0 or pressure <= 1e-9:
                continue
            intensity = min(1.0, pressure / max(float(np.max(state)), 1e-9))
            red = int(105 + 150*intensity)
            green = int(75*(1-intensity))
            colour = f"rgba({red},{green},35,{0.25+0.75*intensity:.2f})"
            # Place a prominent arrowhead halfway along each connection,
            # clear of the node markers. Plotly's arrow tip is at (x, y).
            if is_incoming:
                # Supplier node -> focal buyer: arrow points inward.
                tip_x, tip_y = 0.43*x1, 0.43*y1
                tail_x, tail_y = 0.70*x1, 0.70*y1
            else:
                # Focal supplier -> buyer node: arrow points outward.
                tip_x, tip_y = 0.70*x1, 0.70*y1
                tail_x, tail_y = 0.43*x1, 0.43*y1
            fig.add_annotation(
                x=tip_x, y=tip_y, ax=tail_x, ay=tail_y,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=2.2,
                arrowwidth=3.0+1.5*intensity, arrowcolor=colour,
                standoff=0, startstandoff=0
            )

    ids = [focus] + neighbour_ids
    xs = [positions[i][0] for i in ids]
    ys = [positions[i][1] for i in ids]
    values = [float(state[i]) for i in ids]
    labels_short = [labels[i].replace("_", " · ") for i in ids]
    hover = [
        f"<b>{labels[i]}</b><br>Modelled shock: {state[i]:.2%}<br>" +
        ("Selected focus activity" if i == focus else
         f"Dependency weight to/from focus: {weights.get(i, 0.0):.4%}")
        for i in ids
    ]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        text=labels_short, textposition="top center",
        textfont=dict(size=10, color="#e2e8f0"),
        customdata=ids, hovertext=hover, hoverinfo="text",
        marker=dict(
            size=[34] + [20 + 15*v for v in values[1:]],
            color=values,
            colorscale=[[0, "#64748b"], [0.03, "#fca5a5"],
                        [0.35, "#ef4444"], [1, "#7f1d1d"]],
            cmin=0, cmax=1,
            colorbar=dict(title="Shock<br>intensity"),
            line=dict(width=1.5, color="#334155")
        ),
        name="Activities"
    ))
    fig.update_layout(
        height=590, margin=dict(l=20, r=35, t=35, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[-1.45, 1.45], fixedrange=True),
        yaxis=dict(visible=False, range=[-1.35, 1.35], fixedrange=True,
                   scaleanchor="x", scaleratio=1),
        showlegend=False, clickmode="event+select"
    )
    return fig, ids

st.set_page_config(page_title="2022 Production Network Shock Explorer",page_icon="🌐",layout="wide")
st.title("How a shock travels through global production")
st.caption("A focused, interactive view of the 2022 OECD ICIO supplier–buyer network.")
try:
    load_bar = st.progress(5, text="Loading local 2022 network files…")
    B,node_index,metrics,sector_summary = load_network()
    load_bar.progress(100, text="2022 network data loaded")
    load_bar.empty()
except Exception as exc:
    st.error(f"Could not load the persisted 2022 research network: {exc}")
    st.info("Check that the results_network_analysis folder is beside app.py and contains the four expected 2022 WD files.")
    st.stop()

# Lightweight fingerprint: invalidates cached results if the local adjacency file changes.
network_stat = WD_MATRIX_PATH.stat()
network_version = f"{network_stat.st_size}_{network_stat.st_mtime_ns}"

labels = node_index.node.astype(str).tolist()
label_to_id = {v:i for i,v in enumerate(labels)}
if "source_node" not in st.session_state: st.session_state.source_node = labels[0]
with st.sidebar:
    st.header("Set up a shock")
    severity_pct = st.slider("Initial shock (%)",1,100,40,1)
    rho = st.slider("Recovery per step (%)",1,100,20,1)/100
    eta = st.slider("Transmission multiplier (η)",.1,2.0,1.1,.1)
    steps = st.slider("Number of steps to simulate",1,30,12)
    st.caption("Start with the defaults. Increase the initial shock to see a stronger disruption.")

all_nodes = metrics.copy()
all_nodes["node"] = all_nodes.node.astype(str)
all_nodes["sector"] = all_nodes.node.map(get_sector)
all_nodes["country"] = all_nodes.node.map(get_country)
source_name = st.selectbox("1. Choose the activity hit first",labels,
    index=labels.index(st.session_state.source_node),
    format_func=lambda x:x.replace("_"," · "))
source = label_to_id[source_name]
if st.session_state.get("previous_source_node") != source_name:
    st.session_state.focus_node = source_name
    st.session_state.previous_source_node = source_name
st.session_state.source_node = source_name
if "focus_node" not in st.session_state:
    st.session_state.focus_node = source_name
if st.session_state.get("_focus_widget_sync") != st.session_state.focus_node:
    st.session_state.focus_selector = st.session_state.focus_node
    st.session_state._focus_widget_sync = st.session_state.focus_node
degree = np.asarray(B.getnnz(axis=0)).ravel()+np.asarray(B.getnnz(axis=1)).ravel()
if degree[source] == 0:
    st.warning("This activity has no retained links in the 2022 network (τ = 0.001), so a shock cannot travel from it.")
    sector_strength = all_nodes.groupby("sector",as_index=False).out_strength.sum().sort_values("out_strength",ascending=False)
    top_sectors = sector_strength.head(5).sector.astype(str).tolist()
    fallback_sector = st.selectbox("Explore one of the five most connected sectors",top_sectors)
    fallback_nodes = all_nodes[all_nodes.sector==fallback_sector].sort_values("out_strength",ascending=False).head(5)
    if not fallback_nodes.empty:
        replacement = st.selectbox("Choose an activity in this sector",fallback_nodes.node.tolist())
        if st.button("Use this activity as the shock source"):
            st.session_state.source_node = replacement
            st.rerun()

with st.spinner("Calculating the network spectral radius…"):
    rho_radius = get_spectral_radius(B, network_version)
beta_critical = rho/rho_radius if rho_radius > 0 else float("inf")
beta = eta*beta_critical if np.isfinite(beta_critical) else 0.0
history = run_sis(B, source, severity_pct/100, beta, rho, steps, network_version)

st.markdown("### 2. Watch the shock move")
st.markdown(f"**{source_name.replace('_',' · ')}** is hit first with a **{severity_pct}%** shock. At each step, disruption is transmitted from a supplier to the activities that buy from it. Those activities can then pass exposure to their own buyers.")
step = st.slider("Propagation timeline — move this slider to inspect each step",0,len(history)-1,len(history)-1)
state = history[step]
focus_name = st.selectbox("Zoom into an activity", labels,
    index=labels.index(st.session_state.focus_node),
    format_func=lambda x:x.replace("_"," · "),
    key="focus_selector")
focus = label_to_id[focus_name]
st.session_state.focus_node = focus_name
st.session_state._focus_widget_sync = focus_name
focus_shock = float(state[focus])
c1,c2,c3 = st.columns(3)
c1.metric("Focused neighbours", "Up to 28")
c2.metric("Shock at focused activity", f"{focus_shock:.2%}")
c3.metric("Activities affected in network", f"{int(np.count_nonzero(state>1e-6)):,}")
st.caption("The selected activity is centred. Large arrowheads sit midway along each connection. Arrows pointing toward the centre mean supplier → selected activity; arrows pointing away mean selected activity → buyer. Grey nodes are not currently affected; redder nodes have stronger modelled shocks. Click a neighbour to zoom into it without restarting the original shock.")
fig, displayed_nodes = make_figure(B,labels,state,focus)
st.plotly_chart(
    fig,
    use_container_width=True,
    key="focused_network_click",
    on_select="rerun",
    selection_mode="points",
)
selection = st.session_state.get("focused_network_click", {})
selected_points = selection.get("selection", {}).get("points", [])
if selected_points:
    point = selected_points[-1]
    node_id = point.get("customdata")
    if node_id is None:
        point_index = point.get("point_index")
        if point_index is not None and 0 <= int(point_index) < len(displayed_nodes):
            node_id = displayed_nodes[int(point_index)]
    if node_id is not None and 0 <= int(node_id) < len(labels):
        st.session_state.focus_node = labels[int(node_id)]
        st.rerun()

st.markdown("### 3. What does the model calculate?")
st.markdown("**In plain language:** the selected activity starts with the chosen shock. A buyer is exposed when one or more of its suppliers are shocked. Exposure depends on the supplier-to-buyer dependency weight in the research network. At the same time, recovery reduces the shock remaining at each activity. This repeats one step at a time.")
with st.expander("Show the research equation and data assumptions"):
    st.markdown("**Research Model A (discrete-time SIS):**")
    st.latex(r"d_j(t+1)=\operatorname{clip}_{[0,1]}\left[(1-\rho)d_j(t)+(1-d_j(t))\beta\sum_i B_{ij}d_i(t)\right]")
    st.markdown("- d_j(t) is the modelled shock intensity of activity j, between 0 and 1.\n- B_ij is the research dependency weight from supplier i to buyer j.\n- Bᵀd calculates incoming shock exposure received by each buyer.\n- ρ is the recovery rate; β controls transmission strength.\n- The initial activity starts at the selected shock percentage; all other activities start at zero.\n\n**Important:** this is a modelled propagation index, not a forecast of percentage output loss, a probability of firm failure, or a monetary estimate. This SIS model includes recovery but does not calculate substitution gains.")
    st.markdown("**2022 data:** 4,050 country–industry activities; weighted-directed (WD) network; supplier rows point to buyer columns. Dependency weights are each supplier's share of the buyer's total intermediate purchases. Self-use remains in the denominator, self-edges are excluded, and the research-selected threshold is τ = 0.001.")
    st.caption(f"Network spectral radius: {rho_radius:.6g} · Critical transmission βc: {beta_critical:.6g} · Applied β: {beta:.6g}")

st.markdown("### Most affected activities at this step")
affected = [int(i) for i in np.argsort(state)[::-1] if state[i]>1e-6][:10]
if affected:
    table = pd.DataFrame({"Activity":[labels[i].replace("_"," · ") for i in affected],
        "Shock intensity":[state[i] for i in affected],"Country":[get_country(labels[i]) for i in affected],
        "Industry":[get_sector(labels[i]) for i in affected]})
    st.dataframe(table.style.format({"Shock intensity":"{:.2%}"}),use_container_width=True,hide_index=True)
else:
    st.info("No activity currently has a measurable shock at this step.")
