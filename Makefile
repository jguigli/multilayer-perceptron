all: process train predict

process:
	@cd ./source/ && python3 process.py
train:
	@python3 train.py
predict:
	@python3 predict.py
clean:
	@cd ./data/ && rm X_train X_test y_train y_test

histogram:
	@python3 histogram.py
scatter:
	@python3 scatter_plot.py
pair:
	@python3 pair_plot.py