FROM python:3.10-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean

# Copy and install dependencies
COPY requirements.txt install.sh ./
RUN chmod +x install.sh && ./install.sh

# Copy app code
COPY . .

# Expose port for Render
EXPOSE 10000

# Run using gunicorn
CMD ["gunicorn", "updated_flask_app:app", "--bind", "0.0.0.0:10000"]
