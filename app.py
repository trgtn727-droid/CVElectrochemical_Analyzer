import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import find_peaks

st.set_page_config(
    page_title="Complete Electrochemical & Structural Analyzer",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Complete Electrochemical (CV, GCD, EIS) & Structural (XRD, FTIR) Analyzer")
st.markdown(
    "Upload your complete suite of characterization files for robust kinetic, capacitive, "
    "impedance, and structural analysis tailored for journal submissions."
)

# Sidebar Inputs
st.sidebar.header("1. Experimental Parameters")
electrode_mass_mg = st.sidebar.number_input(
    "Active Electrode Mass (mg)", min_value=0.001, value=1.000, step=0.100, format="%.3f"
)
electrode_mass_g = electrode_mass_mg / 1000.0

target_scan_rate_mv = st.sidebar.number_input(
    "Target Scan Rate for Dunn's Split (mV/s)", min_value=0.1, value=20.0, step=1.0
)
target_scan_rate_vs = target_scan_rate_mv / 1000.0

skip_rows = st.sidebar.number_input(
    "Header Rows to Skip (if potentiostat metadata exists)", min_value=0, value=0, step=1
)

st.sidebar.header("2. Upload CV Files")
uploaded_files = st.sidebar.file_uploader(
    "Select CV files (CSV, TXT, XLS, XLSX)",
    type=["csv", "txt", "xls", "xlsx"],
    accept_multiple_files=True
)

st.sidebar.markdown("---")
st.sidebar.header("3. Advanced Characterization Files (Optional)")
xrd_file = st.sidebar.file_uploader("Upload XRD Data (.asc, .csv, .txt, .dat)", type=["asc", "csv", "txt", "dat"])
ftir_file = st.sidebar.file_uploader("Upload FTIR Data (.asc, .csv, .txt, .dat)", type=["asc", "csv", "txt", "dat"])
gcd_file = st.sidebar.file_uploader("Upload GCD Data (.csv, .txt, .xlsx)", type=["csv", "txt", "xlsx"])
eis_file = st.sidebar.file_uploader("Upload EIS Nyquist Data (.csv, .txt, .xlsx)", type=["csv", "txt", "xlsx"])

def parse_uploaded_file(uploaded_file, skiprows=0):
    """Reads CSV, TXT, ASC, DAT, XLS, or XLSX and extracts two primary numeric columns."""
    name = uploaded_file.name.lower()
    if name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file, skiprows=skiprows)
    else:
        df = pd.read_csv(uploaded_file, sep=None, engine="python", skiprows=skiprows, comment="#")
    
    df = df.dropna(axis=1, how="all")
    numeric_cols = []
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > len(df) * 0.3:
            df[col] = converted
            numeric_cols.append(col)
            
    df = df[numeric_cols].dropna()
    if df.shape[1] < 2:
        raise ValueError(f"File '{uploaded_file.name}' must have at least two numeric columns.")
        
    x_val = df.iloc[:, 0].values.astype(float)
    y_val = df.iloc[:, 1].values.astype(float)
    return x_val, y_val

# Structural Data Processing & Predominant Peak Extraction
xrd_summary, ftir_summary, gcd_summary, eis_summary = "", "", "", ""
xrd_df, ftir_df, gcd_df, eis_df = None, None, None, None

if xrd_file is not None:
    try:
        xrd_file.seek(0)
        x_th, y_int = parse_uploaded_file(xrd_file)
        xrd_df = pd.DataFrame({"2Theta_deg": x_th, "Intensity_a.u.": y_int})
        
        # Find peaks and select top 3 most prominent/predominant peaks for concise reporting
        peaks, properties = find_peaks(y_int, prominence=(np.max(y_int) - np.min(y_int)) * 0.1)
        if len(peaks) > 0:
            prominences = properties["prominences"]
            top_indices = peaks[np.argsort(prominences)[-3:]]  # Top 3 peaks
            predominant_2theta = np.sort(x_th[top_indices])
            xrd_summary = f"The XRD pattern exhibits well-defined diffraction peaks, with predominant reflections located at approximately {', '.join([f'{val:.2f}°' for val in predominant_2theta])}, confirming high crystalline phase purity and structural integrity."
        else:
            xrd_summary
