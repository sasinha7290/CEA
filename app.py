import streamlit as st
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import ttest_ind, f_oneway
from sklearn.metrics import roc_curve, roc_auc_score
import StepMiner  

st.set_page_config(layout="wide")
st.title("🧬 Composite Score and Multi-Group AUC Analyzer")

# ---------- STEP 1: UPLOAD ----------
expr_file = st.file_uploader("Upload Expression File (TSV or CSV)", type=["txt","tsv","csv"])
if expr_file:
    sep = "\t" if expr_file.name.endswith((".tsv",".txt")) else ","
    expr = pd.read_csv(expr_file, sep=sep)
    expr = expr.set_index(expr.columns[0])
    st.success(f"Loaded {expr.shape[0]} features × {expr.shape[1]} samples")

    # ---------- STEP 2: SELECT GROUPS ----------
    st.header("Step 2: Define Sample Groups")
    sample_names = expr.columns.tolist()
    num_groups = st.number_input("Number of Groups", min_value=2, max_value=5, value=2)
    groups = {}
    for i in range(num_groups):
        groups[f"Group_{i+1}"] = st.multiselect(f"Select samples for Group {i+1}", sample_names)

    # ---------- STEP 3: DEFINE GENE SET AND WEIGHTS ----------
    st.header("Step 3: Define Gene Set and Weights")
    gene_input = st.text_area(
        "Enter gene:weight pairs (one per line, comma-separated)",
        "CCL2:1\nCXCL1:1\nIL6:1\nIL8:1\nCCDC88A:-1"
    )
    genes, weights = [], []
    for line in gene_input.strip().splitlines():
        if ":" not in line: continue
        g, w = line.split(":")
        genes.append(g.strip().upper())
        weights.append(float(w))

    # ---------- ANALYSIS FUNCTIONS ----------
    def threshold(values):
        """Adaptive step threshold using StepMiner."""
        arr = np.array(values, dtype=float)
        arr.sort()
        s = StepMiner.fitstep(arr)
        thr = s["threshold"] + 0.5
        return thr

    def compute_composite_score(expr_df, genes, weights):
        expr_df.index = expr_df.index.str.upper()
        samples = expr_df.columns
        composite = np.zeros(len(samples))
        for g, w in zip(genes, weights):
            if g not in expr_df.index:
                continue
            v = expr_df.loc[g].values
            sd = np.std(v)
            if sd <= 0 or not np.isfinite(sd):
                continue
            thr = threshold(v)
            composite += w * (v - thr) / (3 * sd)
        return pd.Series(composite, index=samples, name="CompositeScore")

    # ---------- RUN ANALYSIS ----------
    if st.button("Run Analysis"):
        comp = compute_composite_score(expr, genes, weights)
        results = pd.DataFrame({"Sample": comp.index, "CompositeScore": comp.values})

        # Assign user-defined groups
        label_map = {}
        for gname, slist in groups.items():
            for s in slist:
                label_map[s] = gname
        results["Group"] = results["Sample"].map(label_map)

        st.subheader("Composite Scores")
        st.dataframe(results)

        # ---------- Boxplot ----------
        plt.figure(figsize=(6,4))
        sns.boxplot(x="Group", y="CompositeScore", data=results, palette="Set2")
        sns.stripplot(x="Group", y="CompositeScore", data=results, color="black", alpha=0.6)
        st.pyplot(plt)

        # ---------- Statistics ----------
        unique_groups = results["Group"].dropna().unique()
        if len(unique_groups) == 2:
            g1, g2 = [results[results["Group"]==x]["CompositeScore"] for x in unique_groups]
            stat, pval = ttest_ind(g1, g2)
            st.write(f"**t-test:** p = {pval:.3e}")
        elif len(unique_groups) > 2:
            arrays = [results[results["Group"]==x]["CompositeScore"] for x in unique_groups]
            stat, pval = f_oneway(*arrays)
            st.write(f"**ANOVA:** p = {pval:.3e}")

        # ---------- ROC / AUC ----------
        ref = unique_groups[0]
        auc_summary = []
        for grp in unique_groups[1:]:
            subset = results[results["Group"].isin([ref, grp])]
            if subset["Group"].nunique() < 2:
                continue
            y_true = (subset["Group"] == grp).astype(int)
            y_score = subset["CompositeScore"].values
            auc_val = roc_auc_score(y_true, y_score)
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_summary.append({"Comparison": f"{grp}_vs_{ref}", "AUC": auc_val})

            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
            ax.plot([0,1],[0,1],"--",color="gray")
            ax.legend()
            ax.set_title(f"{grp} vs {ref}")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            st.pyplot(fig)

        if auc_summary:
            st.subheader("AUC Summary")
            st.dataframe(pd.DataFrame(auc_summary))
