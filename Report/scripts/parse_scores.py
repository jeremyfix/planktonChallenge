import pandas as pd
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 2:
    print(f"Usage : {sys.argv[0]} kagglescorepage.html")
    sys.exit(-1)

kaggle_html = sys.argv[1]

num_participants = 13

dfs = pd.read_html(kaggle_html)
fig, ax = plt.subplots()
for i in range(1, num_participants + 1):
    dfs[i]["Date"] = pd.to_datetime(dfs[i]["Date"])
    dfs[i]["Public"] = pd.to_numeric(dfs[i]["Public"], errors="coerce")
    df = dfs[i].sort_values(by="Date", ascending=True)
    df["Public"] = df["Public"].cummax()
    df.plot(x="Date", y="Public", ax=ax, legend=False)
ax.set_title("Scores through time on the public test set")
ax.set_ylabel("Macro F1 score")
ax.set_ylim(0.0, 1)
plt.savefig("../figs/public-test.pdf", bbox_inches="tight")

dfs = pd.read_html(kaggle_html)
fig, ax = plt.subplots()
for i in range(1, num_participants + 1):
    dfs[i]["Date"] = pd.to_datetime(dfs[i]["Date"])
    dfs[i]["Private"] = pd.to_numeric(dfs[i]["Private"], errors="coerce")
    df = dfs[i].sort_values(by="Date", ascending=True)
    df["Private"] = df["Private"].cummax()
    df.plot(x="Date", y="Private", ax=ax, legend=False)
ax.set_title("Scores through time on the public test set")
ax.set_ylabel("Macro F1 score")
ax.set_ylim(0.0, 1)
plt.savefig("../figs/private-test.pdf", bbox_inches="tight")
