# 2022 Interactive Shock Propagation

An interactive Streamlit visualization of the **2022 OECD ICIO weighted-directed production network**, using the exact persisted WD matrix from [International_Trade_Shock_Analysis](https://github.com/AadityaPala0211/International_Trade_Shock_Analysis).

## Run locally

Python 3.10+

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The app downloads and caches the 2022 WD adjacency matrix, node index, node metrics and sector summary from the public research repository. It does not parse the 90 MB raw CSV on each run. The 2019 CSV is intentionally unused.

## Interaction

1. Select a country–industry activity or click a node in the graph.
2. Set initial shock severity, recovery rate, threshold-normalised transmission and propagation steps.
3. Inspect the network at a selected step and the most affected activities.
4. If the selected node has no retained WD links at τ = 0.001, choose from the five strongest connected sectors and the top five activities within that sector.

## Research consistency

- 4,050 country–industry nodes; supplier → buyer direction.
- WD weights are dependency shares P(i,j) = Z(i,j) / total intermediate purchases of buyer j.
- Self-use remains in the denominator; diagonal graph edges are excluded.
- Uses the persisted research network at the analyst-selected threshold τ = 0.001.
- Uses research Model A, with beta = eta × beta-critical and beta-critical = rho / spectral-radius(B).
- The selected node starts at the chosen severity; all other nodes start at zero.

The visualization displays counterfactual model states, **not observed production losses or monetary forecasts**. The SIS model does not estimate substitution gains.
