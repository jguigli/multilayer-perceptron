all: process train predict

process:
	@cd ./source/ && python3 process.py

train:
	@mkdir -p ./saved_parameters
	@mkdir -p ./saved_model
	@mkdir -p ./saved_metrics
	@cd ./source/ && python3 train.py

predict:
	@cd ./source/ && python3 predict.py

clean:
	@cd ./data_sets/ && rm X_train.csv X_validation.csv y_train.csv y_validation.csv
	@cd ./source && rm -rf __pycache__
	@cd ./source/components && rm -rf __pycache__
	@rm -rf ./saved_parameters/*
	@rm -rf ./saved_model/*
	@rm -rf ./saved_metrics/*


fclean: clean
	@rm -rf ./saved_parameters
	@rm -rf ./saved_model
	@rm -rf ./saved_metrics
