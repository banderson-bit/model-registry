FROM ghcr.io/mlflow/mlflow:latest

# Install PostgreSQL driver and other dependencies
RUN pip install psycopg2-binary boto3

# Set environment variables for AWS/MinIO credentials
ENV AWS_ACCESS_KEY_ID=minio
ENV AWS_SECRET_ACCESS_KEY=minio123
ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000

EXPOSE 5000

# Default command
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]