import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(page_title="Elite CRM Data Mining Optimizer", layout="wide")

st.title("Elite CRM Data Mining Optimizer (Platform Starter Demo)")

# --- FILE UPLOAD ---
uploaded_file = st.sidebar.file_uploader("Upload your customer CSV", type=["csv"])
if not uploaded_file:
    st.info("Please upload a customer CSV to get started.")
    st.stop()

df = pd.read_csv(uploaded_file)

# --- COLUMN AUTODETECT ---
patterns = {
    'name': ['name', 'full', 'contact'],
    'first_name': ['first'],
    'last_name': ['last'],
    'email': ['email'],
    'phone': ['phone', 'mobile', 'cell'],
    'visit': ['visit', 'date', 'order', 'lastseen', 'totalvisit'],
    'spend': ['spend', 'amount', 'total', 'revenue', 'price', 'paid', 'sales', 'concession'],
    'product': ['product', 'service', 'item', 'race', 'package'],
}
col_map = {}
for col in df.columns:
    for field, pats in patterns.items():
        if any(p in col.lower() for p in pats):
            col_map[field] = col

with st.expander("Edit column mappings"):
    for key in patterns.keys():
        options = [None] + list(df.columns)
        sel = st.selectbox(
            f"Column for '{key}'", 
            options, 
            index=options.index(col_map.get(key, None)) if col_map.get(key, None) in options else 0
        )
        if sel: col_map[key] = sel

crm_df = pd.DataFrame()
for k, v in col_map.items():
    if v in df.columns:
        crm_df[k] = df[v]

# --- SEGMENTATION ---
from datetime import datetime
if 'visit' in crm_df.columns:
    crm_df['visit'] = pd.to_datetime(crm_df['visit'], errors='coerce')
    crm_df['Recency'] = (datetime.now() - crm_df['visit']).dt.days
if 'email' in crm_df.columns and 'visit' in crm_df.columns:
    freq = crm_df.groupby('email')['visit'].count().rename("Frequency")
    crm_df = crm_df.merge(freq, on='email', how='left')
if 'email' in crm_df.columns and 'spend' in crm_df.columns:
    crm_df['spend'] = pd.to_numeric(crm_df['spend'], errors='coerce')
    mon = crm_df.groupby('email')['spend'].sum().rename("Monetary")
    crm_df = crm_df.merge(mon, on='email', how='left')
crm_df = crm_df.drop_duplicates('email')
feature_cols = [col for col in ['Recency','Frequency','Monetary'] if col in crm_df.columns]
seg_labels = []
if len(feature_cols) >= 2 and crm_df[feature_cols].nunique().sum() > 2:
    X = crm_df[feature_cols].fillna(crm_df[feature_cols].median())
    kmeans = KMeans(n_clusters=min(4, len(crm_df)), n_init=10, random_state=42)
    crm_df['Segment'] = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    for i in range(kmeans.n_clusters):
        rec = centers[i][0] if 'Recency' in feature_cols else 0
        freq = centers[i][1] if 'Frequency' in feature_cols else 0
        mon = centers[i][2] if len(feature_cols) == 3 else 0
        if 'Recency' in feature_cols and rec > np.median(X['Recency']):
            seg_labels.append("At-risk")
        elif freq > np.median(X['Frequency']) and (('Monetary' in feature_cols and mon > np.median(X['Monetary'])) or 'Monetary' not in feature_cols):
            seg_labels.append("VIP")
        elif freq > np.median(X['Frequency']):
            seg_labels.append("Active")
        else:
            seg_labels.append("Regular")
    crm_df['SegmentLabel'] = crm_df['Segment'].apply(lambda x: seg_labels[x])
else:
    crm_df['SegmentLabel'] = "All Customers"
    seg_labels = ["All Customers"]

# --- SIDEBAR NAVIGATION ---
tabs = ["Overview"] + sorted(set(seg_labels))
tab = st.sidebar.radio("Select Segment", tabs)

# --- MAIN UI ---
if tab == "Overview":
    st.header("CRM Overview")
    st.markdown("**Quick summary of all customers, all segments.**")
    st.dataframe(crm_df)
    st.subheader("Segment Breakdown")
    st.dataframe(
        crm_df['SegmentLabel'].value_counts().reset_index().rename(columns={'index':'Segment','SegmentLabel':'Count'})
    )
    fig, ax = plt.subplots()
    crm_df['SegmentLabel'].value_counts().plot.pie(
        autopct='%1.1f%%', ax=ax, startangle=90, colors=plt.cm.Paired.colors[:len(seg_labels)]
    )
    ax.set_ylabel('')
    ax.set_title('Customer Segments')
    st.pyplot(fig)

else:
    st.header(f"{tab} Segment")
    segment_df = crm_df[crm_df['SegmentLabel'] == tab]
    st.write(f"**{len(segment_df)} customers in this segment**")
    st.dataframe(segment_df)
    if 'Recency' in segment_df.columns:
        st.subheader("Recency Distribution (Days Since Last Visit)")
        fig2, ax2 = plt.subplots()
        segment_df['Recency'].dropna().plot.hist(bins=20, ax=ax2)
        ax2.set_xlabel('Days Since Last Visit')
        st.pyplot(fig2)
    if 'Monetary' in segment_df.columns:
        st.subheader("Revenue Distribution (Total Spend)")
        fig3, ax3 = plt.subplots()
        segment_df['Monetary'].dropna().plot.hist(bins=20, ax=ax3)
        ax3.set_xlabel('Total Revenue')
        st.pyplot(fig3)
    st.download_button(
        label=f"Download {tab} segment as CSV",
        data=segment_df.to_csv(index=False).encode(),
        file_name=f"crm_{tab.lower().replace(' ','_')}.csv",
        mime="text/csv"
    )

# --- DOWNLOAD FULL DATA ---
st.sidebar.subheader("Download All Data")
st.sidebar.download_button(
    label="Download Full CRM as CSV",
    data=crm_df.to_csv(index=False).encode(),
    file_name="enriched_crm.csv",
    mime="text/csv"
)

st.sidebar.info("Want AI recommendations, client login, or dashboard upgrades? Just say NEXT!")

