"""
UCLA Admission Predictor - interactive Streamlit app.

Everything updates in real time: predictions recompute instantly as you move
the sliders (no submit button), and the data-exploration charts respond to
the filters. Your applicant profile is overlaid on the dataset so you can
see exactly where you stand.
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="UCLA Admission Predictor", page_icon="🎓",
                   layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))


# ─── Cached loaders ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    models_dir = os.path.join(HERE, "models")
    with open(os.path.join(models_dir, "ucla_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(models_dir, "ucla_mlp_model.pkl"), "rb") as f:
        model = pickle.load(f)
    return scaler, model


@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(HERE, "data", "Admission.csv"))
    df["Strong_Chance"] = (df["Admit_Chance"] >= 0.8).astype(int)
    return df


scaler, model = load_artifacts()
df = load_data()
feature_names = list(scaler.feature_names_in_)


# ─── Sidebar: applicant profile (live - no button) ───────────────────────
st.sidebar.header("🎓 Applicant Profile")
st.sidebar.caption("Predictions update instantly as you adjust the values.")

gre = st.sidebar.slider("GRE Score", 260, 340, 320)
toefl = st.sidebar.slider("TOEFL Score", 60, 120, 105)
unirating = st.sidebar.select_slider("University Rating", [1, 2, 3, 4, 5], value=3)
sop = st.sidebar.slider("SOP Strength", 1.0, 5.0, 3.5, 0.5)
lor = st.sidebar.slider("LOR Strength", 1.0, 5.0, 3.5, 0.5)
cgpa = st.sidebar.slider("CGPA", 5.0, 10.0, 8.5, 0.05)
research = st.sidebar.toggle("Research Experience", value=True)


def build_row():
    """Assemble the model input in the exact training column order."""
    row = dict.fromkeys(feature_names, 0)
    row["GRE_Score"] = gre
    row["TOEFL_Score"] = toefl
    row["SOP"] = sop
    row["LOR"] = lor
    row["CGPA"] = cgpa
    for i in range(1, 6):
        row[f"University_Rating_{i}"] = 1 if unirating == i else 0
    row["Research_1"] = 1 if research else 0
    row["Research_0"] = 0 if research else 1
    return pd.DataFrame([row], columns=feature_names)


X_applicant = build_row()
prob = float(model.predict_proba(scaler.transform(X_applicant))[0, 1])
pred = int(prob >= 0.5)

# ─── Header + live headline metrics ──────────────────────────────────────
st.title("🎓 UCLA Admission Predictor")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Admission Chance", f"{prob:.1%}")
m2.metric("Verdict", "Likely Admit ✅" if pred else "Unlikely ❌")
admitted = df[df["Strong_Chance"] == 1]
m3.metric("Your GRE vs admits", f"{gre}",
          delta=f"{gre - admitted['GRE_Score'].mean():+.0f} vs avg admit")
m4.metric("Your CGPA vs admits", f"{cgpa:.2f}",
          delta=f"{cgpa - admitted['CGPA'].mean():+.2f} vs avg admit")

st.progress(prob, text=f"Predicted probability of a strong admission chance: {prob:.1%}")

tab_predict, tab_explore, tab_model = st.tabs(
    ["🔮 Prediction Breakdown", "📊 Explore the Data", "🧠 Model Performance"])


# ─── Tab 1: prediction breakdown ─────────────────────────────────────────
with tab_predict:
    left, right = st.columns(2)

    with left:
        st.subheader("You vs. the average admitted applicant")
        compare = pd.DataFrame({
            "You": {"GRE": gre, "TOEFL": toefl, "SOP": sop,
                    "LOR": lor, "CGPA": cgpa},
            "Avg admit": {"GRE": admitted["GRE_Score"].mean(),
                          "TOEFL": admitted["TOEFL_Score"].mean(),
                          "SOP": admitted["SOP"].mean(),
                          "LOR": admitted["LOR"].mean(),
                          "CGPA": admitted["CGPA"].mean()},
            "Avg non-admit": {"GRE": df[df.Strong_Chance == 0]["GRE_Score"].mean(),
                              "TOEFL": df[df.Strong_Chance == 0]["TOEFL_Score"].mean(),
                              "SOP": df[df.Strong_Chance == 0]["SOP"].mean(),
                              "LOR": df[df.Strong_Chance == 0]["LOR"].mean(),
                              "CGPA": df[df.Strong_Chance == 0]["CGPA"].mean()},
        }).round(2)
        st.dataframe(compare, width="stretch")

        # normalized profile bars so different scales are comparable
        norm = compare.copy()
        for idx, (lo, hi) in {"GRE": (260, 340), "TOEFL": (60, 120),
                              "SOP": (1, 5), "LOR": (1, 5),
                              "CGPA": (5, 10)}.items():
            norm.loc[idx] = (compare.loc[idx] - lo) / (hi - lo)
        st.bar_chart(norm, height=260)

    with right:
        st.subheader("What moves your chances? (live sensitivity)")
        st.caption("Each line shows your predicted probability as ONE factor "
                   "varies while the others stay at your current values.")

        def sweep(col, values, setter):
            probs = []
            for v in values:
                r = X_applicant.copy()
                setter(r, v)
                probs.append(model.predict_proba(scaler.transform(r))[0, 1])
            return probs

        gre_range = np.arange(280, 341, 5)
        cgpa_range = np.round(np.arange(6.0, 10.01, 0.25), 2)
        sens = pd.DataFrame({
            "GRE sweep": pd.Series(
                sweep("GRE_Score", gre_range,
                      lambda r, v: r.__setitem__("GRE_Score", v)),
                index=(gre_range - 280) / 60),
            "CGPA sweep": pd.Series(
                sweep("CGPA", cgpa_range,
                      lambda r, v: r.__setitem__("CGPA", v)),
                index=(cgpa_range - 6.0) / 4.0),
        })
        st.line_chart(sens, height=260)
        st.caption("x-axis normalized 0-1 across each factor's range "
                   "(GRE 280-340, CGPA 6.0-10.0).")

        flip = X_applicant.copy()
        flip["Research_1"], flip["Research_0"] = (0, 1) if research else (1, 0)
        prob_flip = model.predict_proba(scaler.transform(flip))[0, 1]
        st.info(f"Toggling research experience would change your chance from "
                f"**{prob:.1%}** to **{prob_flip:.1%}**.")


# ─── Tab 2: dataset exploration with live filters ────────────────────────
with tab_explore:
    f1, f2, f3 = st.columns(3)
    rating_filter = f1.multiselect("University Rating", [1, 2, 3, 4, 5],
                                   default=[1, 2, 3, 4, 5])
    research_filter = f2.radio("Research", ["All", "With research", "Without"],
                               horizontal=True)
    gre_min, gre_max = f3.slider("GRE range", 290, 340, (290, 340))

    view = df[df["University_Rating"].isin(rating_filter)]
    if research_filter == "With research":
        view = view[view["Research"] == 1]
    elif research_filter == "Without":
        view = view[view["Research"] == 0]
    view = view[view["GRE_Score"].between(gre_min, gre_max)]

    s1, s2, s3 = st.columns(3)
    s1.metric("Applicants in view", len(view))
    s2.metric("Strong-chance rate",
              f"{view['Strong_Chance'].mean():.0%}" if len(view) else "-")
    s3.metric("Avg CGPA", f"{view['CGPA'].mean():.2f}" if len(view) else "-")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("GRE vs CGPA - you are the star ⭐")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=view, x="GRE_Score", y="CGPA",
                        hue="Strong_Chance", palette={0: "#d62728", 1: "#2ca02c"},
                        alpha=0.6, ax=ax)
        ax.scatter([gre], [cgpa], marker="*", s=600, c="#1f77b4",
                   edgecolors="black", zorder=5, label="You")
        ax.legend(title="Strong chance")
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.subheader("Where you sit in each distribution")
        metric_choice = st.selectbox(
            "Pick a metric", ["GRE_Score", "TOEFL_Score", "CGPA", "SOP", "LOR"])
        your_value = {"GRE_Score": gre, "TOEFL_Score": toefl, "CGPA": cgpa,
                      "SOP": sop, "LOR": lor}[metric_choice]
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(data=view, x=metric_choice, hue="Strong_Chance",
                     palette={0: "#d62728", 1: "#2ca02c"}, alpha=0.5, ax=ax)
        ax.axvline(your_value, color="#1f77b4", lw=3, ls="--")
        ax.text(your_value, ax.get_ylim()[1] * 0.9, "  You", color="#1f77b4",
                fontweight="bold")
        pct = (view[metric_choice] < your_value).mean() if len(view) else 0
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"You score higher than **{pct:.0%}** of applicants "
                   f"currently in view on {metric_choice.replace('_', ' ')}.")

    with st.expander("🔎 Browse the filtered raw data"):
        st.dataframe(view.drop(columns=["Serial_No"]), width="stretch",
                     height=300)
        st.download_button("Download filtered data as CSV",
                           view.to_csv(index=False), "admissions_filtered.csv",
                           "text/csv")


# ─── Tab 3: model performance ────────────────────────────────────────────
with tab_model:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("How well does the model do?")

        @st.cache_data
        def dataset_performance():
            X_all = pd.get_dummies(
                df.drop(columns=["Serial_No", "Admit_Chance", "Strong_Chance"])
                  .astype({"University_Rating": "object", "Research": "object"}),
                columns=["University_Rating", "Research"], dtype="int",
            ).reindex(columns=feature_names, fill_value=0)
            preds = model.predict(scaler.transform(X_all))
            acc = (preds == df["Strong_Chance"]).mean()
            cm = pd.crosstab(df["Strong_Chance"], preds,
                             rownames=["Actual"], colnames=["Predicted"])
            return acc, cm

        acc, cm = dataset_performance()
        st.metric("Accuracy on full dataset", f"{acc:.1%}")
        st.caption("Held-out test accuracy during training: **90.0%** "
                   "(see train_model.py).")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title("Confusion matrix (full dataset)")
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.subheader("Training loss curve")
        loss_path = os.path.join(HERE, "assets", "loss_curve.png")
        if os.path.exists(loss_path):
            st.image(loss_path, width="stretch")
        elif hasattr(model, "loss_curve_"):
            st.line_chart(pd.Series(model.loss_curve_, name="loss"))
        st.caption("MLPClassifier: two hidden layers (3, 3), min-max scaled "
                   "features, batch size 50.")

st.divider()
st.caption("Educational demo - not intended for real admissions decisions.")
