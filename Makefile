date := $(shell date +'%s')
req_file_name = aml-requirements-$(date).txt

# Create datastores and register datasets
register-data:
	# This is not the ideal way to attach a datastore. Usually, this would be done before you are submitting jobs.
	# However, given the demo nature of this repo, a link to the demo dataset was included here. 
	az ml datastore attach-blob -c diabetes -a publicmldatasc -n diabetes --sas-token "?si=DiabetesReadOnly&sv=2019-10-10&sr=c&sig=cz9P%2B1V1eC6FvDIKBQNmA5nWqbsGfkqzdPTTYmiidfg%3D"
	az ml dataset register -f aml/datasets/diabetes_table.json --skip-validation
	az ml dataset register -f aml/datasets/diabetes_files.json --skip-validation

create-compute:
	az ml computetarget create amlcompute --max-nodes 5 --idle-seconds-before-scaledown 600 -n cpu-cluster -s standard_ds15_v2

config: register-data create-compute

env: environment

# Build the AML environment
environment:
	az ml environment register -d aml/env
	python aml/utils/build_env.py --name MLFlow-XGBoost

# Build or pull the AML environment locally
environment-local:
	az ml environment register -d aml/env
	python aml/utils/build_env.py --name MLFlow-XGBoost --local

train:
	az ml run submit-script --path code/train -c cloud_run.yaml

# Update the AML SDK to the latest version
upgrade-aml-sdk:
	pip list -o | grep azureml | awk '{ print $$1"=="$$3}' > $(req_file_name)
	pip install -r $(req_file_name)
	rm $(req_file_name)

lint:
	flake8