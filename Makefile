python := python3

define venvWrapper
	{\
	. bin/activate; \
	$1; \
	}
endef


install:
	@{ \
		echo "Setting up..."; \
		python3 -m venv .; \
		. bin/activate; \
		if [ -f requirements.txt ]; then \
			pip install -r requirements.txt; \
			echo "Installing dependencies...DONE"; \
		fi; \
	}

freeze:
	$(call venvWrapper, pip freeze > requirements.txt)

all: install process train predict

process:
	@$(call venvWrapper, cd ./source/ && python3 process.py)

train:
	@mkdir -p ./saved_parameters
	@mkdir -p ./saved_model
	@mkdir -p ./saved_metrics
	@$(call venvWrapper, cd ./source/ && python3 train.py)

predict:
	@$(call venvWrapper, cd ./source/ && python3 predict.py)

clean:
	@cd ./data_sets/ && rm X_train.csv X_validation.csv y_train.csv y_validation.csv
	@cd ./source && rm -rf __pycache__
	@cd ./source/components && rm -rf __pycache__
	@rm -rf ./saved_parameters/*
	@rm -rf ./saved_model/*
	@rm -rf ./saved_metrics/*


fclean: clean
	@rm -rf bin/ include/ lib/ lib64 pyvenv.cfg share/
	@rm -rf ./saved_parameters
	@rm -rf ./saved_model
	@rm -rf ./saved_metrics

phony: install freeze process train predict