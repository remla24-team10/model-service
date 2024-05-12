# Use a base image that supports Python 3.11
FROM python:3.11-slim

WORKDIR /model-service

# Copy the files into the container
COPY poetry.lock pyproject.toml encoder.pkl tokenizer.pkl trained_model.keras /model_service/app.py /model-service/

# Install Poetry and dependencies
RUN pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry lock --no-update \
    && poetry install --no-dev

# Set the command to run the Flask app
CMD ["python", "app.py"]