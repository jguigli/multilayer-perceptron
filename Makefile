all: process train predict

process:
	@cd ./source/ && python3 process.py
train:
	@cd ./source/ && python3 train.py
predict:
	@cd ./source/ && python3 predict.py
clean:
	@cd ./data_sets/ && rm X_train.csv X_validation.csv y_train.csv y_validation.csv