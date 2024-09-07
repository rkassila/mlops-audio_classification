# Makefile

# Default target
all: setup

# Create a virtual environment, copy config, and install dependencies
setup:
	pyenv install -s 3.10.6
	pyenv virtualenv 3.10.6 audio_class_env
	pyenv local audio_class_env
	pip install --upgrade pip
	pip install -r requirements.txt
	cp config/gcs_config.example.yaml config/gcs_config.yaml
	echo "Setup finished : env created and activated, requirements installed"

# Run the main program
run:
	python main.py

# Clean up the project directory
clean:
	rm -rf .venv
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +

# Installing flake8
install-tools:
	python -m pip install flake8

# Lint the code using flake8
lint:
	python -m flake8 scripts/

.PHONY: all setup run lint clean install-tools
