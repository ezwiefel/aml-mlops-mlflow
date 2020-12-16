date := $(shell date +'%s')
req_file_name = aml-requirements-$(date).txt

# Create datastores and register datasets
register-data:
	az ml datastore attach-blob -c diabetes -a publicmldatasc -n diabetes --sas-token "?si=DiabetesReadOnly&sv=2019-10-10&sr=c&sig=cz9P%2B1V1eC6FvDIKBQNmA5nWqbsGfkqzdPTTYmiidfg%3D"
	az ml dataset register -f aml/datasets/diabetes-table.json
	az ml dataset register -f aml/datasets/diabetes-files.json

create-compute:
	az ml computetarget create amlcompute --max-nodes --idle-seconds-before-scaledown 600 -n cpu-cluster -s standard_ds15_v2

config: register-data create-compute

# Build the AML environment
environment:
	az ml environment register -d aml/env
	python aml/utils/build_env.py --name MLFlow-XGBoost

# Build or pull the AML environment locally
environment-local:
	az ml environment register -d aml/env
	python aml/utils/build_env.py --name MLFlow-XGBoost --local

train: environment
	az ml run submit-script -c cpu-cluster

# Update the AML SDK to the latest version
upgrade-aml-sdk:
	pip list -o | grep azureml | awk '{ print $$1"=="$$3}' > $(req_file_name)
	pip install -r $(req_file_name)
	rm $(req_file_name)