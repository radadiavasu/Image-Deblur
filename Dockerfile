# Use official Python 3.10 image
FROM python:3.10-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system packages
RUN apt-get update && \
    apt-get install -y git wget libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean

# Copy and install dependencies
COPY requirements.txt install.sh ./
RUN chmod +x install.sh && ./install.sh

# Copy the rest of the app
COPY . .

# Expose for Render
EXPOSE 10000

# Run the app
CMD ["gunicorn", "updated_flask_app:app", "--bind", "0.0.0.0:10000"]
