# Use official Python 3.10 image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies for git (needed by pip)
RUN apt-get update && apt-get install -y git && apt-get clean

# Copy requirement files and install dependencies
COPY requirements.txt install.sh ./
RUN chmod +x install.sh && ./install.sh

# Copy all remaining project files into the container
COPY . .

# Expose port (Render automatically maps to it)
EXPOSE 10000

# Start the app with gunicorn
CMD ["gunicorn", "updated_flask_app:app", "--bind", "0.0.0.0:10000"]
