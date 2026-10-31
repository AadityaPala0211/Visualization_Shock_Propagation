from __future__ import annotations
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
        raise ValueError("The research network must contain exactly 4,050 aligned nodes.")
    index = index.sort_values("node_id").reset_index(drop=True)
    metrics = metrics.sort_values("node_id").reset_index(drop=True)
    if not np.array_equal(index.node_id.to_numpy(), metrics.node_id.to_numpy()):
        raise ValueError("Node index and metrics have inconsistent ordering.")
    matrix.setdiag(0); matrix.eliminate_zeros()
    return matrix, index, metrics, sectors

def spectral_radius(B):
    if B.nnz == 0: return 0.0
    return float(abs(eigs(B, k=1, which="LM", return_eigenvectors=False, tol=1e-5, maxiter=3000)[0]))

def run_sis(B, source, severity, beta, rho, steps):
    state = np.zeros(B.shape[0], dtype=float); state[source] = severity
    history = [state.copy()]; BT = B.T.tocsr()
    for _ in range(steps):
        force = np.asarray(BT @ state).ravel()
        state = np.clip((1-rho)*state + (1-state)*beta*force, 0, 1)
        history.append(state.copy())
        if state.max() < 1e-8: break
    return np.asarray(history)

def sector(label): return label.split("_", 1)[1] if "_" in label else label
def country(label): return label.split("_", 1)[0] if "_" in label else ""

def make_figure(B, labels, state, source, limit=145):
    undirected = B + B.T
    neighbors = undirected.indices[undirected.indptr[source]:undirected.indptr[source+1]]
    candidates = set(neighbors.tolist()) | set(np.argsort(state)[-limit:].tolist())
    candidates.discard(source)
    ranked = sorted(candidates, key=lambda i: state[i], reverse=True)
    chosen = [source] + ranked[:limit-1]
    sub = B[chosen, :][:, chosen].tocsr()
    graph = nx.from_scipy_sparse_array(sub, create_using=nx.DiGraph)
    pos = nx.spring_layout(graph, seed=21, iterations=35, weight="weight")
    coo = sub.tocoo(); ex=[]; ey=[]
    for i,j in zip(coo.row,coo.col):
        if i == j: continue
        ex += [pos[i][0],pos[j][0],None]; ey += [pos[i][1],pos[j][1],None]
    labels_here = [labels[i] for i in chosen]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex,y=ey,mode="lines",line=dict(width=.7,color="rgba(120,135,155,.3)"),hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[pos[i][0] for i in range(len(chosen))],
        y=[pos[i][1] for i in range(len(chosen))],mode="markers+text",
        text=labels_here,textposition="top center",textfont=dict(size=8),
        customdata=chosen,hovertext=[f"{labels[i]}<br>Shock state: {state[i]:.3f}" for i in chosen],
        hoverinfo="text",marker=dict(size=[14+19*state[i] for i in chosen],
        color=[state[i] for i in chosen],colorscale="YlOrRd",cmin=0,cmax=1,
        colorbar=dict(title="Shock state"),line=dict(width=1,color="#334155"))))
    fig.add_trace(go.Scatter(x=[pos[0][0]],y=[pos[0][1]],mode="markers",
        marker=dict(size=25,color="#2563eb",symbol="circle-open",line=dict(width=3)),hoverinfo="skip"))
    fig.update_layout(height=650,margin=dict(l=8,r=8,t=15,b=8),paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",xaxis=dict(visible=False),yaxis=dict(visible=False),
        showlegend=False,clickmode="event+select")
    return fig

st.set_page_config(page_title="ICIO 2022 Shock Propagation",page_icon="🌐",layout="wide")
st.title("International Production Network · 2022")
st.caption("Interactive supplier-to-buyer shock propagation · OECD ICIO · 4,050 country–industry activities")
try:
    B, node_index, metrics, sector_summary = load_network()
except Exception as exc:
    st.error(f"Could not load the persisted 2022 research network: {exc}")
    st.info("Check the connection and public research repository.")
    st.stop()

labels = node_index.node.astype(str).tolist()
label_to_id = {v:i for i,v in enumerate(labels)}
if "source_node" not in st.session_state: st.session_state.source_node = labels[0]
with st.sidebar:
    st.header("Shock controls")
    severity_pct = st.slider("Initial shock severity",1,100,40,1,format="%d%%")
    rho = st.slider("Recovery rate (ρ)",.01,1.0,.20,.01)
    eta = st.slider("Transmission level (η = β / βc)",.1,2.0,1.1,.1)
    steps = st.slider("Propagation steps",1,40,12)
    st.caption("The selected percentage is the initial shock state of the clicked activity.")

all_nodes = metrics.copy()
all_nodes["node"] = all_nodes.node.astype(str)
all_nodes["sector"] = all_nodes.node.map(sector)
all_nodes["country"] = all_nodes.node.map(country)
source_name = st.selectbox("Choose an activity (or click a node below)",labels,
    index=labels.index(st.session_state.source_node),format_func=lambda x:x.replace("_"," · "))
source = label_to_id[source_name]; st.session_state.source_node = source_name
degree = np.asarray(B.getnnz(axis=0)).ravel() + np.asarray(B.getnnz(axis=1)).ravel()
if degree[source] == 0:
    st.warning(f"{source_name} has no retained links at the research threshold τ = 0.001.")
    sector_strength = all_nodes.groupby("sector",as_index=False).out_strength.sum().sort_values("out_strength",ascending=False)
    top_sectors = sector_strength.head(5).sector.astype(str).tolist()
    fallback_sector = st.selectbox("Explore one of the five strongest connected sectors",top_sectors)
    fallback_nodes = all_nodes[all_nodes.sector == fallback_sector].sort_values("out_strength",ascending=False).head(5)
    if not fallback_nodes.empty:
        replacement = st.selectbox("Top five activities in this sector",fallback_nodes.node.tolist())
        if st.button("Use selected activity"):
            st.session_state.source_node = replacement; st.rerun()

rho_radius = spectral_radius(B)
beta_c = rho/rho_radius if rho_radius > 0 else float("inf")
beta = eta*beta_c if np.isfinite(beta_c) else 0.0
history = run_sis(B,source,severity_pct/100,beta,rho,steps)
step = st.slider("View propagation step",0,len(history)-1,len(history)-1)
state = history[step]
c1,c2,c3,c4 = st.columns(4)
c1.metric("Activities",f"{B.shape[0]:,}"); c2.metric("Directed dependencies",f"{B.nnz:,}")
c3.metric("Activities affected",f"{int(np.count_nonzero(state>1e-6)):,}")
c4.metric("Peak shock state",f"{state.max():.1%}")
st.subheader("Network propagation")
st.caption("Arrows run from supplier to buyer. The source is ringed in blue; warmer colours indicate a larger modelled shock state.")
fig = make_figure(B,labels,state,source)
clicked = plotly_events(fig,click_event=True,select_event=False,hover_event=False,override_height=650,key="network_click")
if clicked:
    point = clicked[0].get("customdata")
    if point is not None and labels[int(point)] != st.session_state.source_node:
        st.session_state.source_node = labels[int(point)]; st.rerun()

left,right = st.columns([1.15,.85])
with left:
    st.subheader("Most affected activities")
    ids = [int(i) for i in np.argsort(state)[::-1] if state[i]>1e-6][:15]
    table = pd.DataFrame({"Activity":[labels[i] for i in ids],"Shock state":[state[i] for i in ids],
        "Country":[country(labels[i]) for i in ids],"Industry":[sector(labels[i]) for i in ids]})
    st.dataframe(table.style.format({"Shock state":"{:.2%}"}),use_container_width=True,hide_index=True)
with right:
    st.subheader("Propagation over time")
    trajectory = go.Figure()
    trajectory.add_trace(go.Scatter(y=history.mean(axis=1),mode="lines+markers",name="Mean state"))
    trajectory.add_trace(go.Scatter(y=history.max(axis=1),mode="lines+markers",name="Maximum state"))
    trajectory.update_layout(height=310,margin=dict(l=8,r=8,t=10,b=8),xaxis_title="Step",
        yaxis_title="Shock state",yaxis=dict(range=[0,1]))
    st.plotly_chart(trajectory,use_container_width=True)

with st.expander("Model and data details"):
    st.markdown(r"""
**Network:** persisted 2022 weighted-directed (WD) matrix; supplier rows point to buyer columns. Each weight is the supplier's share of the buyer's total intermediate purchases. Self-use remains in the denominator, self-edges are removed, and the analyst-selected threshold is **τ = 0.001**.

**Research Model A:**

\[
d_j(t+1)=\operatorname{clip}_{[0,1]}\left[(1-\rho)d_j(t)+(1-d_j(t))\beta\sum_i B_{ij}d_i(t)\right]
\]

The selected node starts at the chosen severity; all other nodes start at zero. Transmission is set to \(\beta=\eta\beta_c\), with \(\beta_c=\rho/\lambda_{max}(B)\).

These are counterfactual model states, not observed production losses or dollar forecasts. The SIS model represents propagation and recovery; it does not estimate substitution gains.
""")
    st.write(f"Spectral radius: {rho_radius:.6g} · Critical transmission βc: {beta_c:.6g} · Applied β: {beta:.6g}")
    st.write("Source: International_Trade_Shock_Analysis / results_network_analysis / empirical / 2022 / WD.")
