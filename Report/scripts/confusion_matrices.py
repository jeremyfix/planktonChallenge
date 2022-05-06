# coding: utf-8

import sys
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def confusion_matrix(num_classes, df):
    crosstab = pd.crosstab(df["label"], df["predicted"], normalize="index")
    # Some indices might be missing in the "predicted"
    # because the predictor may never predict that class
    indices = list(range(num_classes))
    crosstab = crosstab.reindex(indices, axis=1).fillna(0)
    return crosstab


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} gt_labels.csv csvdir")
        print(f"e.g. python3 {sys.argv[0]} gt_all.csv participants_csv")
        sys.exit(-1)

    # Merge the csvs and the ground truth
    gt_labelfilename = sys.argv[1]
    csvdir = pathlib.Path(sys.argv[2])

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

    class_stats = df["label"].value_counts().to_frame("counts")
    num_classes = len(class_stats)

    # Compute the confusion matrices, for each participant
    for team in teams:
        print(f"Evaluating the confusion matrix for the team {team}")
        subdf = df[["label", team]].rename(columns={team: "predicted"})
        print(subdf.head())
        confusion_mtx = confusion_matrix(num_classes, subdf)
        print(confusion_mtx)
        f = plt.figure(figsize=(15, 15))
        sns.heatmap(confusion_mtx, cmap="Blues", vmin=0, vmax=1)
        # plt.imshow(confusion_mtx, vmin=0, vmax=1, aspect="equal")

        ax = plt.gca()
        ax.set_xlim(0, num_classes)
        ax.set_ylim(num_classes, 0)
        plt.grid()
        plt.savefig(f"../figs/confusion_matrix_{team}.png", bbox_inches="tight")
        plt.close(f)
