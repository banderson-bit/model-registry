# Data Science Model Registry

A system for deploying, managing, and utilizing data science models via Kafka. This application allows models to be registered, versioned, and distributed to applications that need to make predictions.

## Architecture

The system consists of several components:

1. **Model Registry** - A service for storing and retrieving trained ML models
2. **MLflow Integration** - For experiment tracking, model versioning and registry

## Requirements

- Python 3.8+
- Podman and Podman Compose for MLflow infrastructure
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd ds_models
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Start the Kafka and MLFlow infrastructure:

```bash
cd infrastructure
podman-compose up -d
```

This will start:
- PostgreSQL for MLflow backend storage
- MinIO as S3-compatible storage for model artifacts
- MLflow server on http://localhost:5000
- Zookeeper and Kafka for messaging

## Components

### MLflow Integration

The system includes MLflow integration for advanced model management:

- **Model Training and Pickling**: Models are trained and saved as pickle files
- **MLflow Registry**: Models are registered in MLflow for versioning and tracking
- **Experiment Tracking**: Training metrics and parameters are tracked in MLflow
- **Model Serving**: Models can be loaded and served directly from MLflow

### Example Model: Treasury Price Prediction

The system includes an example model that predicts monthly prices based on US Treasury printing data. The model uses four features:

- `treasury_print_volume` - Volume of treasury prints
- `interest_rate` - Current interest rate
- `inflation_rate` - Current inflation rate
- `previous_month_price` - Previous month's price

### Kafka Topics

- `model.registry` - Notifications about model updates
- `treasury.data` - Input data events with treasury information
- `signals.topic` - Output signals with price predictions

## Usage

### 1. Train and Pickle a Model

```bash
python models/train_and_pickle_model.py
```

This will:
- Create a TreasuryPriceModel
- Generate synthetic training data
- Train the model
- Evaluate the model
- Save it as a pickle file to models/pickles/treasury_price_model.pkl

### 2. Register the Model in MLflow

```bash
python models/register_model_in_mlflow.py
```

This will:
- Load the pickled model
- Create an MLflow experiment
- Log model parameters and metrics
- Create a model wrapper compatible with MLflow's pyfunc interface
- Register the model in MLflow under the name 'treasury-price-model'

### 3. Run the Traditional Integration Test

To test the entire system end-to-end using the traditional registry:

```bash
python tests/test_model_api_server.py 
```

## MLflow UI

Access the MLflow UI to see registered models, experiments, and runs:

```
http://localhost:5000
```

The UI provides:
- Model versions and stages
- Experiment tracking
- Parameter and metric visualization
- Model artifact management

## Integration with GitHub Actions

To integrate this model registry with GitHub Actions:

1. Create a workflow file in your GitHub repository (`.github/workflows/train-and-deploy-model.yml`):

```yaml
name: Train and Deploy Model

on:
  push:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 1'  # Weekly on Monday

jobs:
  train-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        
    - name: Train and register model
      run: |
        python models/train_and_register_model.py
```

## Future Enhancements and Design Alternatives

### Storage Alternatives
- **Cloud Object Storage**: Instead of local file storage, consider using AWS S3, Google Cloud Storage, or Azure Blob Storage for model storage.
- **Database Storage**: For more complex metadata and queries, consider using MongoDB, PostgreSQL or specialized model tracking databases.

### Model Registry Alternatives
- **MLflow**: A more comprehensive platform for tracking experiments, packaging models, and deploying them.
- **Weights & Biases**: For experiment tracking, model versioning, and dataset versioning.
- **DVC (Data Version Control)**: For more robust version control of both models and data.

### Message Queue Alternatives
- **RabbitMQ**: For more complex routing patterns.
- **AWS SQS/SNS**: For cloud-native deployments.
- **Google Pub/Sub**: For Google Cloud deployments.

### Monitoring and Observability
- Add monitoring for model drift and performance degradation
- Implement A/B testing capability for new models
- Add data validation to ensure input data quality

### Security Enhancements
- Model encryption at rest
- Authentication and authorization for model access
- Input data validation and sanitization

## Benefits of the MLflow Integration

1. **Reproducible Model Training**: The training process is standardized and can be reproduced consistently
2. **Model Versioning**: MLflow tracks all model versions and their performance metrics
3. **Centralized Model Registry**: Models are stored in a central location accessible to all applications
4. **Deployment Ready**: The solution integrates with Kafka for real-time prediction on streaming data
5. **Testable**: The integration test verifies the entire workflow from training to prediction

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.