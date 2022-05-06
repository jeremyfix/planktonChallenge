# Analysis of the competitors submissions 

## Generating the report

To generate the report, you just need to 

```
make paper.pdf
```

You may want to regenerate some data used in the report :

### Generate the examples collage in appendix

**Requirements**: the training dataset

```
rm -rf collage && make collage
```

### Generate the class distribution statistics


**Requirements**: the training dataset and the public/private test set csv

```
rm figs/stats.pdf && make clastats
```

### Generate the evolution of metric through time

**Requirements** : the kaggle score page html

```
rm figs/public-test.pdf figs/private-test.pdf && make scorestime
```

### Generate the F1 scores distributions

**Requirements**: the participants csv, the class names csv and public/private test set csv

```
rm figs/class_f1_scores_per_team.pdf figs/class_f1_scores_per_class.pdf ./figs/table_best_predicted.tex
./figs/table_worst_predicted.tex &&  make classteamf1
```

### Generate the confusion matrices


**Requirements**: the participants csv, the class names csv and public/private test set csv

```
rm -rf figs/confusion* && make confusion
```
