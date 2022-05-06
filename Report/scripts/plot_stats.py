# coding: utf-8
import sys
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
    print(f"Usage {sys.argv[0]} traindir testcsv")
    sys.exit(-1)

traindir = sys.argv[1]
testcsv = sys.argv[2]

stats = pd.read_csv(testcsv, sep=",", header=1)
stats.columns = ["filename", "class", "set"]

private_test = stats[stats["set"] == "Private"]
private_class_distrib = private_test["class"].value_counts()
private_class_distrib = pd.DataFrame(
    {"class": private_class_distrib.index, "Private test": private_class_distrib.values}
)
private_class_distrib = private_class_distrib.set_index("class")

public_test = stats[stats["set"] == "Public"]
public_class_distrib = public_test["class"].value_counts()
public_class_distrib = pd.DataFrame(
    {"class": public_class_distrib.index, "Public test": public_class_distrib.values}
)
public_class_distrib = public_class_distrib.set_index("class")
print(public_class_distrib)

# Collect the training set stats and class names
class_names = {}

trainset = []
for path in os.scandir(traindir):
    id_name = path.name.split("_")
    class_names[int(id_name[0])] = " ".join(id_name[1:])
    l = os.listdir(path)
    trainset.append((int(id_name[0]), len(l)))
trainset = pd.DataFrame(trainset, columns=["class", "Train"])
trainset = trainset.set_index("class")

df = public_class_distrib.join(private_class_distrib).join(trainset)
print(df.cumsum())
df.plot.bar(figsize=(15, 5))

plt.xticks(horizontalalignment="center")
plt.yscale("log")
plt.ylabel("Number of samples")
plt.tight_layout()
# plt.show()
plt.savefig("stats.pdf")

# Generate the latex table
df.reset_index()
with open("stat_table.tex", "w") as f:
    f.write("\\begin{tabular}{|c|r|c|c|c|}\n")
    f.write("Index & Name & Train & Public & Private\\\\\n\hline")
    for idx, (index, row) in enumerate(df.iterrows()):
        f.write(
            f"{index}  & {class_names[index]} & {row['Train']}& {row['Public test']}& {row['Private test']}\\\\\n"
        )
        if idx == 43:
            f.write("\hline\n")
            f.write("\\end{tabular}\n\n")
            f.write("\\begin{tabular}{|c|r|c|c|c|}\n")
            f.write("Index & Name & Train & Public & Private\\\\\n\hline")
    f.write("\hline\n")
    f.write("\\end{tabular}\n")
