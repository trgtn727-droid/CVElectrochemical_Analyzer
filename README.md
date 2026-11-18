# ⚡ Electrochemical Cyclic Voltammetry (CV) Modeling & Kinetics App

A web-based electrochemical data analysis application built with **Streamlit**, **NumPy**, **SciPy**, and **Pandas**. Supports batch uploading of multi-scan-rate Cyclic Voltammetry files in **`.csv`**, **`.txt`**, **`.xls`**, and **`.xlsx`** formats.

## Key Capabilities
- **Multi-Format Batch Ingestion:** Automatically parses `.csv`, `.txt`, `.xls`, and `.xlsx` potentiostat exports.
- **Automated Redox Peak Detection:** Extracts anodic/cathodic peak potentials ($E_{pa}, E_{pc}$), peak currents ($i_{pa}, i_{pc}$), peak separation ($\Delta E_p$), and $|i_{pa}/i_{pc}|$ ratios.
- **Specific Capacitance ($C_s$) & Rate Capability:** Integrates closed CV loop areas ($\int I(V)\,dV$) to compute $C_s$ ($\text{F g}^{-1}$) across all scan rates and evaluates capacitance retention.
- **Kinetic Mechanism ($b$-Value Analysis):** Evaluates $i(V) = a v^b$ via $\log(i)$ vs. $\log(v)$ regression to classify EDLC ($b \approx 1.0$), pseudocapacitive ($0.5 < b < 1.0$), or diffusion-controlled battery behavior ($b \approx 0.5$).
- **Dunn's Method Current Deconvolution:** Separates total current into surface-controlled capacitive ($k_1 v$) and diffusion-controlled ($k_2 v^{0.5}$) contributions.
- **Publication-Ready Export:** Generates 300 DPI multi-panel figures, CSV summary tables, and literature benchmark reports in a single downloadable `.zip` bundle.

## Quick Start (Local)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud via GitHub
1. Push `app.py`, `requirements.txt`, and `README.md` to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your GitHub repository, set the main file path to `app.py`, and click **Deploy**.
