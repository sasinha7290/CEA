# app.py
import sys, os

# Ensure local imports (like StepMiner) work
sys.path.append(os.path.dirname(__file__))

# ---- Safe imports with Streamlit error fallback ----
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError as e:
    import streamlit as st
    st.error(f"Missing package: {e.name}. "
             "Make sure 'requirements.txt' includes matplotlib and seaborn, then redeploy.")
    st.stop()

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway
from sklearn.metrics import roc_curve, roc_auc_score

# ---- Import StepMiner safely ----
try:
    import StepMiner
except ModuleNotFoundError:
    st.warning("⚠️ StepMiner.py not found in this folder. Please ensure it is uploaded.")
    st.stop()

# ---- Streamlit UI ----
st.title("🧬 Composite Expression Analyzer (CEA)")
st.markdown("""
Upload your expression file and define gene sets with weights.
The app will calculate composite scores, visualize group differences, and compute AUC values.
""")

# ---- Upload expression data ----
expr_file = st.file_uploader("Upload Expression File (TSV/CSV)", type=["txt", "csv", "tsv"])
group_file = st.file_uploader("Upload Group File (2 columns: Sample, Group)", type=["txt", "csv", "tsv"])

if expr_file and group_file:
    # Load data
    sep = "\t" if expr_file.name.endswith(".tsv") or expr_file.name.endswith(".txt") else ","
    expr = pd.read_csv(expr_file, sep=sep, index_col=0)
    groups = pd.read_csv(group_file, sep=sep, header=None, names=["Sample", "Group"])

    # Convert gene names to uppercase
    expr.index = expr.index.str.upper()
    groups["Sample"] = groups["Sample"].astype(str)

    # Show data previews
    st.subheader("Expression Matrix (Top 5)")
    st.dataframe(expr.head())
    st.subheader("Group Mapping (Top 5)")
    st.dataframe(groups.head())

    # ---- User gene set input ----
    gene_input = st.text_area("Enter Gene List (comma-separated):", "CCL2, CXCL1, IL6, CXCL8")
    weight_input = st.text_area("Enter Weights (comma-separated, same length as gene list):", "1, 1, 1, 1")

    if st.button("Compute Composite Score"):
        genes = [g.strip().upper() for g in gene_input.split(",")]
        weights = np.array([float(x) for x in weight_input.split(",")])

        # Filter valid genes
        genes_present = [g for g in genes if g in expr.index]
        if not genes_present:
            st.error("None of the entered genes found in the expression matrix.")
            st.stop()

        subset = expr.loc[genes_present]
        weighted_expr = subset.T.dot(weights[:len(genes_present)])
        scores = pd.DataFrame({"Sample": expr.columns, "CompositeScore": weighted_expr.values})

        # Merge with groups
        df = scores.merge(groups, on="Sample", how="left").dropna(subset=["Group"])

        # ---- Visualization ----
        st.subheader("Composite Score Distribution by Group")
        plt.figure(figsize=(6, 4))
        sns.violinplot(x="Group", y="CompositeScore", data=df, inner="box", palette="Set2")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)

        # ---- Statistics ----
        unique_groups = df["Group"].unique()
        st.markdown("### Statistical Comparison Between Groups")
        if len(unique_groups) == 2:
            g1, g2 = [df[df["Group"] == g]["CompositeScore"] for g in unique_groups]
            tstat, pval = ttest_ind(g1, g2, equal_var=False)
            st.write(f"**T-test (unpaired):** p = {pval:.3e}")
        else:
            data_groups = [df[df["Group"] == g]["CompositeScore"] for g in unique_groups]
            fstat, pval = f_oneway(*data_groups)
            st.write(f"**ANOVA:** p = {pval:.3e}")

        # ---- AUC calculation ----
        ref_group = st.selectbox("Select Reference Group for AUC Comparison", unique_groups)
        auc_results = []
        for grp in unique_groups:
            if grp == ref_group:
                continue
            y_true = (df["Group"] == grp).astype(int)
            y_score = df["CompositeScore"]
            auc_val = roc_auc_score(y_true, y_score)
            auc_results.append((grp, auc_val))
        if auc_results:
            st.subheader("AUC Results (vs Reference Group)")
            auc_df = pd.DataFrame(auc_results, columns=["Compared Group", "AUC"])
            st.dataframe(auc_df)

            # Plot ROC curves
            plt.figure(figsize=(5, 5))
            for grp, auc_val in auc_results:
                y_true = (df["Group"] == grp).astype(int)
                y_score = df["CompositeScore"]
                fpr, tpr, _ = roc_curve(y_true, y_score)
                plt.plot(fpr, tpr, label=f"{grp} (AUC={auc_val:.2f})")
            plt.plot([0, 1], [0, 1], "--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.tight_layout()
            st.pyplot(plt)
