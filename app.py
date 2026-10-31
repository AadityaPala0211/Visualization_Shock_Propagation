from __future__ import annotations
from collections import deque
from io import BytesIO
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy import sparse
from scipy.sparse.linalg import eigs
from streamlit_plotly_events import plotly_events

RESEARCH_RAW = "https://raw.githubusercontent.com/AadityaPala0211/International_Trade_Shock_Analysis/main/results_network_analysis"
WD_MATRIX_URL = f"{RESEARCH_RAW}/empirical/2022/WD/adjacency.npz"
NODE_INDEX_URL = f"{RESEARCH_RAW}/metadata/node_index.csv"
NODE_METRICS_URL = f"{RESEARCH_RAW}/empirical/2022/WD/node_metrics.csv.gz"
SECTOR_SUMMARY_URL = f"{RESEARCH_RAW}/empirical/2022/WD/sector_summary.csv"

@st.cache_data(show_spinner="Loading the 2022 production network…")
def load_network():
    session = requests.Session()
    response = session.get(WD_MATRIX_URL, timeout=90); response.raise_for_status()
    matrix = sparse.load_npz(BytesIO(response.content)).tocsr()
    response = session.get(NODE_INDEX_URL, timeout=60); response.raise_for_status()
    index = pd.read_csv(BytesIO(response.content))
    response = session.get(NODE_METRICS_URL, timeout=60); response.raise_for_status()
    metrics = pd.read_csv(BytesIO(response.content), compression="gzip")
    response = session.get(SECTOR_SUMMARY_URL, timeout=60); response.raise_for_status()
    sectors = pd.read_csv(BytesIO(response.content))
    if matrix.shape != (4050, 4050) or len(index) != 4050:
        raise ValueError("Expected the aligned 4,050-node 2022 research network.")
    index = index.sort_values("node_id").reset_index(drop=True)
    metrics = metrics.sort_values("node_id").reset_index(drop=True)
    if not np.array_equal(index.node_id.to_numpy(), metrics.node_id.to_numpy()):
        raise ValueError("Node index and metrics have inconsistent ordering.")
    matrix.setdiag(0); matrix.eliminate_zeros()
    return matrix, index, metrics, sectors

@st.cache_data(show_spinner=False)
def get_spectral_radius(matrix):
    if matrix.nnz == 0: return 0.0
    return float(abs(eigs(matrix, k=1, which="LM", return_eigenvectors=False, tol=1e-5, maxiter=3000)[0]))

def run_sis(B, source, severity, beta, rho, steps):
    state = np.zeros(B.shape[0], dtype=float)
    state[source] = severity
    history = [state.copy()]
    BT = B.T.tocsr()
    for _ in range(steps):
        exposure = np.asarray(BT @ state).ravel()
        state = np.clip((1-rho)*state + (1-state)*beta*exposure, 0, 1)
        history.append(state.copy())
        if state.max() < 1e-8: break
    return np.asarray(history)

def get_sector(label): return label.split("_", 1)[1] if "_" in label else label
def get_country(label): return label.split("_", 1)[0] if "_" in label else ""

def propagation_subset(B, source, state, max_nodes=48, max_hops=3):
    """Show only a small source-reachable outgoing neighbourhood."""
    selected = {int(source)}
    frontier = [int(source)]
    for _ in range(max_hops):
        candidates = {}
        for node in frontier:
            row = B.getrow(node)
            for target, weight in zip(row.indices, row.data):
                target = int(target)
                if target not in selected:
                    score = float(weight) * (0.25 + float(state[target]))
                    candidates[target] = max(candidates.get(target, 0.0), score)
        if not candidates: break
        room = max_nodes - len(selected)
        frontier = sorted(candidates, key=candidates.get, reverse=True)[:max(0, room)]
        selected.update(frontier)
        if len(selected) >= max_nodes: break
    return [source] + sorted(selected - {source}, key=lambda i: state[i], reverse=True)

def make_figure(B, labels, state, source, max_nodes=48):
    chosen = propagation_subset(B, source, state, max_nodes=max_nodes)
    sub = B[chosen, :][:, chosen].tocsr()
    graph = nx.from_scipy_sparse_array(sub, create_using=nx.DiGraph)
    pos = nx.spring_layout(graph, seed=17, k=0.9/np.sqrt(len(chosen)), iterations=55, weight="weight")
    coo = sub.tocoo()
    all_edges = [(int(i), int(j), float(w)) for i,j,w in zip(coo.row,coo.col,coo.data) if i != j]
    active_edges = [(i,j,w) for i,j,w in all_edges if state[chosen[i]] > 1e-7]
    pressures = [state[chosen[i]]*w for i,j,w in active_edges]
    max_pressure = max(pressures, default=0.0)
    fig = go.Figure()
    ex, ey = [], []
    for i,j,w in all_edges:
        ex += [pos[i][0],pos[j][0],None]; ey += [pos[i][1],pos[j][1],None]
    fig.add_trace(go.Scatter(x=ex,y=ey,mode="lines",
        line=dict(width=.8,color="rgba(130,145,165,.20)"),hoverinfo="skip",name="Possible route"))
    # Directional arrows show current supplier-to-buyer shock pressure.
    for (i,j,w), pressure in zip(active_edges,pressures):
        intensity = pressure/max_pressure if max_pressure else 0.0
        alpha = .12 + .86*(intensity**.55)
        colour = f"rgba(255,{int(175-150*intensity)},{int(70-55*intensity)},{alpha:.3f})"
        fig.add_annotation(x=pos[j][0],y=pos[j][1],ax=pos[i][0],ay=pos[i][1],
            xref="x",yref="y",axref="x",ayref="y",showarrow=True,arrowhead=3,
            arrowsize=1.0,arrowwidth=1.2+1.8*intensity,arrowcolor=colour,
            standoff=9,startstandoff=10)
    node_state = [float(state[i]) for i in chosen]
    node_labels = [labels[i].replace("_"," · ") for i in chosen]
    hover = [f"<b>{labels[i]}</b><br>Shock intensity: {state[i]:.1%}<br>{'Initial shock source' if i == source else 'Exposed activity'}" for i in chosen]
    fig.add_trace(go.Scatter(x=[pos[i][0] for i in range(len(chosen))],
        y=[pos[i][1] for i in range(len(chosen))],mode="markers+text",
        text=node_labels,textposition="top center",textfont=dict(size=9,color="#cbd5e1"),
        customdata=chosen,hovertext=hover,hoverinfo="text",
        marker=dict(size=[17+18*x for x in node_state],color=node_state,
        colorscale=[[0,"#dbeafe"],[.25,"#facc15"],[.65,"#f97316"],[1,"#b91c1c"]],
        cmin=0,cmax=1,colorbar=dict(title="Shock<br>intensity"),
        line=dict(width=1,color="#334155")),name="Activities"))
    fig.add_trace(go.Scatter(x=[pos[0][0]],y=[pos[0][1]],mode="markers",
        marker=dict(size=30,color="#38bdf8",symbol="circle-open",line=dict(width=3)),
        hoverinfo="skip",showlegend=False))
    fig.update_layout(height=610,margin=dict(l=8,r=8,t=8,b=8),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),yaxis=dict(visible=False),showlegend=False,
        clickmode="event+select")
    return fig

st.set_page_config(page_title="2022 Production Network Shock Explorer",page_icon="🌐",layout="wide")
st.title("How a shock travels through global production")
st.caption("A focused, interactive view of the 2022 OECD ICIO supplier–buyer network.")
try:
    B,node_index,metrics,sector_summary = load_network()
except Exception as exc:
    st.error(f"Could not load the persisted 2022 research network: {exc}")
    st.info("Check the internet connection and public research repository.")
    st.stop()

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
st.session_state.source_node = source_name
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

rho_radius = get_spectral_radius(B)
beta_critical = rho/rho_radius if rho_radius > 0 else float("inf")
beta = eta*beta_critical if np.isfinite(beta_critical) else 0.0
history = run_sis(B,source,severity_pct/100,beta,rho,steps)

st.markdown("### 2. Watch the shock move")
st.markdown(f"**{source_name.replace('_',' · ')}** is hit first with a **{severity_pct}%** shock. At each step, disruption is transmitted from a supplier to the activities that buy from it. Those activities can then pass exposure to their own buyers.")
step = st.slider("Propagation timeline — move this slider to inspect each step",0,len(history)-1,len(history)-1)
state = history[step]
c1,c2,c3 = st.columns(3)
c1.metric("Activities in graph","Up to 48",help="Only a small source-reachable neighbourhood is shown, not all 4,050 activities.")
c2.metric("Activities with a shock",f"{int(np.count_nonzero(state>1e-6)):,}")
c3.metric("Strongest remaining shock",f"{state.max():.1%}")
st.caption("Arrows point from supplier to buyer. Bright, opaque arrows carry stronger shock pressure; faint arrows carry weaker pressure. Arrow intensity fades as the shock weakens downstream. The blue ring marks the original source.")
fig = make_figure(B,labels,state,source)
clicked = plotly_events(fig,click_event=True,select_event=False,hover_event=False,override_height=610,key="focused_network_click")
if clicked:
    point = clicked[0].get("customdata")
    if point is not None and labels[int(point)] != st.session_state.source_node:
        st.session_state.source_node = labels[int(point)]
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
