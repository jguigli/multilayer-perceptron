all: process train predict

process:
	@python3 process.py
train:
	@python3 train.py
predict:
	@python3 predict.py
clean:
	@rm thetas.csv

histogram:
	@python3 histogram.py
scatter:
	@python3 scatter_plot.py
pair:
	@python3 pair_plot.py