# 2022 Production Network Shock Explorer

A focused interactive Streamlit visualization of the **2022 OECD ICIO weighted-directed production network**, using the persisted empirical network from [International_Trade_Shock_Analysis](https://github.com/AadityaPala0211/International_Trade_Shock_Analysis).

## Run locally

Python 3.10+

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The app downloads and caches the 2022 WD adjacency matrix, node index, node metrics and sector summary. The 2019 data are not used.

## How to use

1. Select the country–industry activity hit by the initial shock.
2. Set the initial shock percentage, recovery rate, transmission multiplier and simulation horizon.
3. Move the propagation timeline slider to inspect shock intensity at each step.
4. Click another activity in the graph to make it the new initial shock source.
5. If the chosen activity is isolated at the research threshold, choose one of the five strongest connected sectors and one of the top five activities in that sector.

## Reading the graph

The graph displays at most 48 activities in the selected source's outgoing, source-reachable neighbourhood, not the full 4,050-node network. Arrows point from supplier to buyer. Bright, opaque arrows indicate stronger current shock pressure; faint arrows indicate weaker pressure. Arrow intensity fades downstream as the modelled shock weakens. The blue ring marks the initial shock source.

## Research consistency and interpretation

- 4,050 country–industry activities; supplier-to-buyer orientation.
- WD weights are dependency shares: P(i,j) = Z(i,j) / total intermediate purchases of buyer j.
- Self-use remains in the denominator; diagonal graph edges are excluded.
- Uses the persisted 2022 WD network at the analyst-selected threshold τ = 0.001.
- Uses research Model A (deterministic mass-action/bilinear SIS); exposure received by buyer j is (Bᵀd)j.
- Critical transmission is βc = ρ / λmax(B); applied transmission is β = η × βc.

The initial shock percentage is a starting **model intensity**, not an observed output reduction. Results are counterfactual propagation indices, not calibrated production-loss or monetary forecasts. The SIS model represents propagation and recovery; it does not estimate substitution gains.
