# coding: utf-8

import sys
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} gt_labels.csv csvdir class_names.csv")
        print(f"e.g. python3 {sys.argv[0]} gt_all.csv participants_csv class_names.csv")
        sys.exit(-1)

    gt_labelfilename = sys.argv[1]
    csvdir = pathlib.Path(sys.argv[2])
    classnames = list(map(lambda x: x.rstrip(), open(sys.argv[3], "r").readlines()))
    print(classnames)

    df = pd.read_csv(
        gt_labelfilename, index_col=0, usecols=["imgname", "label", "Usage"]
    )

    # Now we fill in the submissions of the participants
    teams = []
    for f in csvdir.glob("*.csv"):
        team_name = f.name[:-4].split("_")[1]
        submission_df = pd.read_csv(f, index_col=0)
        submission_df = submission_df.rename(columns={"label": team_name})
        df = pd.concat([df, submission_df], axis=1, join="inner")
        teams.append(team_name)

    print(f"I collected the stats about the following {len(teams)} teams : {teams}")

    # If necessary, split on the board type (public or private)
    # private_set = df[df["Usage"] == "Private"]
    # public_set = df[df["Usage"] == "Public"]

    # For info, count the number of class values per class
    class_stats = df["label"].value_counts().to_frame("counts")
    num_classes = len(class_stats)
    print(f"Class counts for the {num_classes} classes : \n {class_stats}")

    # Measure the class F1 scores for every teams and every class
    for team in teams:
        pred_labels = df[team]
        true_labels = df["label"]
        # for every class, we count the TP, FN, FP
        # 1/F1 = 1/recall + 1/precision
        # precision = TP / (TP + FP)
        # recall = TP / (TP + FN)
        f1s = {"index": [], team: []}
        for cid in range(num_classes):
            pred_labels_cid = (pred_labels == cid).to_numpy().astype("int")
            true_labels_cid = (true_labels == cid).to_numpy().astype("int")
            # For debug, show the number of samples to be predicted as class cid
            # print(f"For class {cid} : {pred_labels_cid.sum()}")
            tp = np.dot(pred_labels_cid, true_labels_cid)
            fp = np.dot(pred_labels_cid, 1 - true_labels_cid)
            fn = np.dot(1 - pred_labels_cid, true_labels_cid)
            if tp + fp == 0:
                precision = 0
            else:
                precision = tp / (tp + fp)
            if tp + fn == 0:
                recall = 0
            recall = tp / (tp + fn)
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2.0 * precision * recall / (precision + recall)

            f1s["index"].append(cid)
            f1s[team].append(f1)

        f1df = pd.DataFrame.from_dict(f1s).set_index("index")
        class_stats = pd.concat([class_stats, f1df], axis=1, join="inner")

    # Sort the teams by their macro F1 : whoaou pandas
    sorted_teams = (
        class_stats.mean()[teams].sort_values(ascending=False).index.to_list()
    )
    # print(class_stats.describe())
    class_stats = class_stats.sort_index()

    class_stats.to_csv("class_stats.csv")

    # class_stats = class_stats.reset_index()
    # sns.boxplot(data=class_stats[teams])
    sns.catplot(data=class_stats[sorted_teams])
    plt.xlabel("Team name")
    plt.ylabel("Class F1 scores")
    plt.xticks(rotation=45, ha="right")
    plt.savefig("../figs/class_f1_scores_per_team.pdf", bbox_inches="tight")

    print(class_stats[teams].T.head())

    plt.figure(figsize=(20, 5))
    sns.catplot(data=class_stats[teams].T[range(num_classes)], height=5, aspect=3)
    plt.xlabel("Class")
    plt.ylabel("Class F1 scores")
    plt.xticks(rotation=45, ha="center")
    plt.savefig("../figs/class_f1_scores_per_class.pdf", bbox_inches="tight")

    # Get the best predicted classes and the worst ones
    stats = class_stats[teams].T.describe().T
    best_predicted = stats[(stats["min"] > 0.8)].index.to_list()
    best_predicted_f1 = stats.T[best_predicted].T["min"]
    worst_predicted = stats[(stats["max"] < 0.6)].index.to_list()
    worst_predicted_f1 = stats.T[worst_predicted].T["max"]
    print(f"The index of the most difficult to predict : {worst_predicted}")
    print(f"The max F1 over all the teams being : \n{worst_predicted_f1}")
    print(f"The index of the easiest to predict: {best_predicted}")
    print(f"The min F1 over all the teams being :\n {best_predicted_f1}")

    # Generate and save the latex tables to be included in the paper
    with open("../figs/table_best_predicted.tex", "w") as f:
        f.write(
            """
\\begin{table}
\\begin{tabular}{|l|r|}
Class names & Minimal F1 over all the teams \\\\
        \\hline
"""
        )
        f.write(
            "\\\\ \n".join(
                f"{cls} {classnames[cls]} & {f1:.2f}"
                for cls, f1 in zip(best_predicted, best_predicted_f1)
            )
        )
        f.write(
            """
\\end{tabular}
\\caption{\\label{table:best_predicted} The classes for which the minimal F1 score obtained by all the teams is above
$0.8$.}
\\end{table}
                """
        )
    with open("../figs/table_worst_predicted.tex", "w") as f:
        f.write(
            """
\\begin{table}
\\begin{tabular}{|l|r|}
Class names & Maximal F1 over all the teams \\\\
        \\hline
"""
        )
        f.write(
            "\\\\ \n".join(
                f"{cls} {classnames[cls]} & {f1:.2f}"
                for cls, f1 in zip(worst_predicted, worst_predicted_f1)
            )
        )
        f.write(
            """
\\end{tabular}
\\caption{\\label{table:worst_predicted} The classes for which the maximal F1 score obtained by all the teams is below $0.6$.}
\\end{table}
                """
        )
