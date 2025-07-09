# Use Python 3.10
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable real-time output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies for pip/git/OpenCV
RUN apt-get update && \
    apt-get install -y git libgl1-mesa-glx && \
    apt-get clean

# Copy and install dependencies
COPY requirements.txt install.sh ./
RUN chmod +x install.sh && ./install.sh

# Copy project files
COPY . .

# Expose the port for Render
EXPOSE 10000

# Start the app using gunicorn
CMD ["gunicorn", "updated_flask_app:app", "--bind", "0.0.0.0:10000"]
