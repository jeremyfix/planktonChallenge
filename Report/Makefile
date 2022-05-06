
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

clastats: ./figs/stats.pdf ./figs/stat_table.tex
figs/stats.pdf: 
	test -s $(DEFAULT_DATAPATH) || { $(error Dataset not available for collage preparation) }
	test -s $(DEFAULT_TESTCSV) || { $(error Test csv file not available) }
	cd scripts && python3 plot_stats.py $(DEFAULT_DATAPATH) $(DEFAULT_TESTCSV)

scorestime: ./figs/public-test.pdf ./figs/private-test.pdf
figs/public-test.pdf: ./data/scoresKaggle.html
	cd scripts && python3 parse_scores.py ../data/scoresKaggle.html


classteamf1: ./figs/class_f1_scores_per_team.pdf ./figs/class_f1_scores_per_class.pdf ./figs/table_best_predicted.tex ./figs/table_worst_predicted.tex 
figs/class_f1_scores_per_team.pdf:./scripts/$(DEFAULT_TESTCSV) ./scripts/$(DEFAULT_PARTICIPANTS_CSV) ./scripts/$(DEFAULT_CLASSNAMES)
	cd scripts && python3 plot_best_submissions.py $(DEFAULT_TESTCSV) $(DEFAULT_PARTICIPANTS_CSV) $(DEFAULT_CLASSNAMES)

confusion: ./figs/confusion_matrix_spongebob.png
figs/confusion_matrix_spongebob.png: ./scripts/$(DEFAULT_TESTCSV) ./scripts/$(DEFAULT_PARTICIPANTS_CSV)  
	cd scripts && python3 confusion_matrices.py $(DEFAULT_TESTCSV) $(DEFAULT_PARTICIPANTS_CSV)  


paper.pdf: paper.tex collage clastats scorestime classteamf1 confusion
	latexmk -pdf paper.tex

