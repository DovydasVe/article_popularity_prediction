import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def make_top_3(model, X_test_sets, y_test_sets, regression=False):
    top3_list = []

    for i, X_test in enumerate(X_test_sets):
        if regression:
            preds = model.predict(X_test)
            
        else:
            preds = model.predict_proba(X_test)[:, 1]

        true_vals = y_test_sets[i]
        order = np.argsort(preds)[::-1][:3]

        top3_df = pd.DataFrame({
            "true_label": true_vals.iloc[order].values,
            "score": preds[order]
        })
        top3_list.append(top3_df)

    return top3_list


def fetch_top_predictions(top3_list, X_test_sets, y_test_sets, y_raw_test_sets):
    hit_list = []
    for i, (top3_df, _, _, _) in enumerate(
        zip(top3_list, X_test_sets, y_test_sets, y_raw_test_sets), 1
    ):
        hits = top3_df["true_label"].sum()
        hit_list.append(hits)

    return hit_list


def visualise_gridsearch(cv_results_df, param_x, param_y, param_split):
    for val in sorted(cv_results_df[param_split].unique()):
        subset = cv_results_df[cv_results_df[param_split] == val]

        pivot_table = subset.pivot_table(
            values="mean_test_score",
            index=param_x,
            columns=param_y
        )

        plt.figure(figsize=(6,4))
        sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")
        plt.title(f"Mean CV Score ({param_split}={val})")
        plt.ylabel(param_x)
        plt.xlabel(param_y)
        plt.show()
