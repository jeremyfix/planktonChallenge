DEFAULT_DATAPATH=/mounts/Datasets3/ChallengeDeep/train
DEFAULT_TESTCSV=../data/gt_all.csv
DEFAULT_PARTICIPANTS_CSV=../data/participants_csv
DEFAULT_CLASSNAMES=../data/class_names.csv

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

scorestime: ./data/scoresKaggle.html
	cd scripts && python3 parse_scores.py ../data/scoresKaggle.html


classteamf1:
	cd scripts && python3 plot_best_submissions.py $(DEFAULT_TESTCSV) $(DEFAULT_PARTICIPANTS_CSV) $(DEFAULT_CLASSNAMES)



paper.pdf: paper.tex collage clastats scorestime classteamf1
	latexmk -pdf paper.tex

