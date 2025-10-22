#!/usr/bin/env python3
# ===========================================================
#  Composite Score and Multi-Group AUC Analyzer (CEA)
#  Author: Saptarshi Sinha
#  Description: Streamlit web app for computing weighted
#  composite expression scores, statistical group tests, and AUC
# ===========================================================

import sys, os
sys.path.append(os.path.dirname(__file__))  # ensures StepMiner.py is importable

import streamlit as st
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import ttest_ind, f_oneway
from sklearn.metrics import roc_curve, roc_auc_score
import StepMiner  # local file, must be in same directory

st.set_page_config(layout="wide", page_title="CEA: Composite Expression Analyzer")

st.title("🧬 Composite Expression Analyzer (CEA)")

# -----------------------------------------------------------
# STEP 1: Upload Expression File
# -----------------------------------------------------------
expr_file = st.file_uploader("Upload Expression Matrix (TSV or CSV)", type=["txt","tsv","csv"])
if expr_file:
    sep = "\t" if expr_file.name.endswith((".tsv",".txt")) else ","
    expr = pd.read_csv(expr_file, sep=sep)
    expr = expr.set_index(expr.columns[0])
    expr.index = expr.index.str.upper()
    st.success(f"Loaded {expr.shape[0]} genes × {expr.shape[1]} samples")

    # -----------------------------------------------------------
    # STEP 2: Define Groups
    # -----------------------------------------------------------
    st.header("Step 2: Define Sample Groups")
    samples = expr.columns.tolist()
    num_groups = st.number_input("Number of groups", min_value=2, max_value=8, value=2)
    groups = {}
    for i in range(num_groups):
        groups[f"Group_{i+1}"] = st.multiselect(f"Select samples for Group {i+1}", samples)

    # -----------------------------------------------------------
    # STEP 3: Define Gene Set and Weights
    # -----------------------------------------------------------
    st.header("Step 3: Define Gene Set and Weights")
    gene_input = st.text_area(
        "Enter gene:weight pairs (comma-separated, one per line)",
        "CCL2:1\nCXCL1:1\nIL6:1\nIL8:1"
    )

    genes, weights = [], []
    for line in gene_input.strip().splitlines():
        if ":" not in line: continue
        g, w = line.split(":")
        genes.append(g.strip().upper())
        weights.append(float(w))

    # -----------------------------------------------------------
    # HELPER: StepMiner threshold-based composite score
    # -----------------------------------------------------------
    def compute_step_threshold(values):
        arr = np.array(values, dtype=float)
        arr.sort()
        s = StepMiner.fitstep(arr)
        return s["threshold"] + 0.5  # consistent with StepMiner usage

    def compute_composite_score(expr_df, genes, weights):
        expr_df.index = expr_df.index.str.upper()
        composite = pd.Series(0.0, index=expr_df.columns)
        for g, w in zip(genes, weights):
            if g not in expr_df.index:
                continue
            v = expr_df.loc[g].values
            sd = np.std(v)
            if sd <= 0 or not np.isfinite(sd):
                continue
            thr = compute_step_threshold(v)
            composite += w * (v - thr) / (3 * sd)
        return composite

    # -----------------------------------------------------------
    # STEP 4: Run Analysis
    # -----------------------------------------------------------
    if st.button("🚀 Run Analysis"):
        scores = compute_composite_score(expr, genes, weights)
        df = pd.DataFrame({"Sample": scores.index, "CompositeScore": scores.values})

        # assign user-defined groups
        label_map = {}
        for gname, slist in groups.items():
            for s in slist:
                label_map[s] = gname
        df["Group"] = df["Sample"].map(label_map)
        df = df.dropna(subset=["Group"])

        st.subheader("📊 Composite Scores")
        st.dataframe(df)

        # -----------------------------------------------------------
        # STEP 5: Box Plot & Statistical Tests
        # -----------------------------------------------------------
        plt.figure(figsize=(7,5))
        sns.boxplot(x="Group", y="CompositeScore", data=df, palette="Set2")
        sns.stripplot(x="Group", y="CompositeScore", data=df, color="black", alpha=0.6)
        st.pyplot(plt)

        unique_groups = df["Group"].unique()
        if len(unique_groups) == 2:
            g1, g2 = [df[df["Group"] == x]["CompositeScore"] for x in unique_groups]
            stat, pval = ttest_ind(g1, g2)
            st.write(f"**t-test** between {unique_groups[0]} and {unique_groups[1]}: p = {pval:.3e}")
        elif len(unique_groups) > 2:
            arrays = [df[df["Group"] == g]["CompositeScore"] for g in unique_groups]
            stat, pval = f_oneway(*arrays)
            st.write(f"**ANOVA** across {len(unique_groups)} groups: p = {pval:.3e}")

        # -----------------------------------------------------------
        # STEP 6: Multi-Group AUC vs Fixed Reference
        # -----------------------------------------------------------
        ref_group = st.selectbox("Select reference group for AUC comparison", unique_groups)
        auc_summary = []

        for grp in unique_groups:
            if grp == ref_group: continue
            subset = df[df["Group"].isin([ref_group, grp])]
            if subset["Group"].nunique() < 2:
                continue
            y_true = (subset["Group"] == grp).astype(int)
            y_score = subset["CompositeScore"].values
            auc_val = roc_auc_score(y_true, y_score)
            fpr, tpr, _ = roc_curve(y_true, y_score)

            auc_summary.append({"Comparison": f"{grp}_vs_{ref_group}", "AUC": auc_val})

            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
            ax.plot([0, 1], [0, 1], "--", color="gray")
            ax.legend()
            ax.set_title(f"{grp} vs {ref_group}")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            st.pyplot(fig)

        if auc_summary:
            st.subheader("📈 AUC Summary")
            st.dataframe(pd.DataFrame(auc_summary))

        # -----------------------------------------------------------
        # STEP 7: Download Results
        # -----------------------------------------------------------
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Composite Scores", data=csv, file_name="CompositeScores.csv")

