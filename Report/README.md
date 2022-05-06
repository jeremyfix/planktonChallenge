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

### Generate 



