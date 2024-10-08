# MlOps Audio Classification

## Description

### Dataset Importation:
Check for dataset version in PostgreSQL, fetch or skip the download, store it in GCS.
### Data Preparation:
Extract features (e.g., MFCCs or spectrograms), split into training/validation sets, check for already-prepared data in GCS.
### Model Training:
Load pre-trained model, fine-tune on the dataset with YAML hyperparameters, track with MLflow, and store in GCS.
### Deployment and Monitoring:
Serve model via FastAPI, monitor performance with Prometheus/Grafana, track and log data/model drift, and handle alerts and rollbacks.


## Technology Stack

- **Python**
- **Hugging Face `datasets`** OK
- **Google Cloud Storage (GCS)** OK
- **PostgreSQL on Google Cloud SQL** OK
- **MLflow** OK
- **FastAPI** OK
- **Prometheus** OK
- **Grafana** OK

## Running Prometheus and Grafana with Docker

This guide provides commands to run Prometheus and Grafana using Docker.

### Commands to Run Prometheus and Grafana

1. **Pull the Docker Images**

   First, pull the latest Docker images for Prometheus and Grafana:

```bash
   docker pull prom/prometheus:latest
   docker pull grafana/grafana:latest
```

### Run Prometheus
```bash
docker run -d --name prometheus -p 9090:9090 -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus:latest
```
### Run Grafana
```bash
docker run -d --name grafana -p 3000:3000 grafana/grafana:latest
```

### Access the Services
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (default username: admin, default password: admin)

### Stop and remove Containers
```bash
docker stop prometheus grafana
docker rm prometheus grafana
```

## Setup and Environmental Requirements

Before running the program, ensure the following are installed:

- Python 3.10.6
- Google Cloud SDK (for GCS operations)

You should have a PostgreSQL database running and a GCS bucket available for uploads.

### Configuration

The program is configured using YAML files:

- **config/datasets_config.yaml:** Contains configurations for datasets to be processed, such as the dataset name, table name, and description.
- **config/gcs_config.yaml:** Contains GCS credentials and bucket configurations.


### Execution Instructions

You can run the pipeline manually or use `make` for automation.
Download the files then follow the instructions below.

### Manual Execution

To set up your configuration:

```bash
cp config/gcs_config.example.yaml config/gcs_config.yaml
```

#### Installing Dependencies

You can install the required Python dependencies manually using:

```bash
pip install -r requirements.txt
```

Ensure your PostgreSQL server is running and your GCS credentials are set up in `gcs_config.yaml` with a configured bucket.

Run the pipeline using:

```bash
python main.py
```

### Execution with `make`

To automate the setup and execution, you can use the provided `Makefile`. The `Makefile` allows you to easily set up your environment, run the program, and clean up.

- **Setup the environment:**

  This creates a virtual environment, installs dependencies, and copies the example config:

```bash
  make setup
```

- **Run the program:**

  Executes the pipeline:

```bash
  make run
```

- **Clean up:**

  Removes the virtual environment and temporary files:

```bash
  make clean
```
