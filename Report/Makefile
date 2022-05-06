DEFAULT_DATAPATH=/mounts/Datasets3/ChallengeDeep/train
DEFAULT_TESTCSV=../data/gt_all.csv

all:paper.pdf

collage/examples.tex:
	test -s $(DEFAULT_DATAPATH) || { $(error Dataset not available for collage preparation) }
	@echo "Collage preparation"
	cd scripts && python3 collage.py $(DEFAULT_DATAPATH) ../collage
collage: collage/examples.tex


clastats: figs/stats.pdf
	test -s $(DEFAULT_DATAPATH) || { $(error Dataset not available for collage preparation) }
	test -s $(DEFAULT_TESTCSV) || { $(error Test csv file not available) }
	cd scripts && python3 plot_stats.py $(DEFAULT_DATAPATH) $(DEFAULT_TESTCSV)



paper.pdf: paper.tex collage clastats
	latexmk -pdf paper.tex

