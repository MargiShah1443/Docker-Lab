# dockerfile
# Docker ML Lab – Train a Breast Cancer classifier inside a container

# 1. Base image
FROM python:3.9-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy source code into the container
COPY src/ .

# 4. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Default command – run the training script
CMD ["python", "main.py"]
