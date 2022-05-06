#!/usr/bin/env python3

import PIL
import sys
import pathlib
import random
import matplotlib.pyplot as plt
import numpy as np


def make_collage(traindir, outputdir, N=10):
    dirs = [d for d in traindir.glob("*") if d.is_dir()]
    outputdir.mkdir()
    # Make the tex file to be included
    texfile = open(outputdir / "examples.tex", "w")

    # Let us sort the images by their class label
    dirs = sorted(dirs, key=lambda d: d.parts[-1])

    for d in dirs:
        # Get some random samples for this class
        print(f"Processing {d}")
        files = list(d.glob("*.jpg"))
        random.shuffle(files)

        fig, axes = plt.subplots(1, N)
        for ax, filename in zip(axes, files):
            img = PIL.Image.open(filename, "r")
            img = np.asarray(img)
            ax.imshow(img, cmap="gray")
            ax.axis("off")
        classname = d.parts[-1]
        plt.tight_layout()
        # plt.suptitle(classname)
        plt.savefig(outputdir / f"{classname}.jpg", bbox_inches="tight")
        plt.close(fig)
        local_path = outputdir.parts[-1] + "/" + f"{classname}.jpg"
        classname = " ".join(classname.split("_")[1:])
        texfile.write(
            "\\begin{figure}[h]\n"
            f"\\includegraphics[width=\\columnwidth]{{{local_path}}}"
            f"\\caption{{Samples from the class {classname} }}\n"
            "\\end{figure}\n"
        )
    texfile.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage : {sys.argv[0]} root_traindir outputdir")
        sys.exit(-1)

    make_collage(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
