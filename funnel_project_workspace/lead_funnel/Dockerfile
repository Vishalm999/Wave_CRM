# Use official Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /new_lead_funnel

# Copy dependencies file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port for FastAPI
EXPOSE 8080

# Run FastAPI using uvicorn
CMD ["uvicorn", "new_lead_funnel:app", "--host", "0.0.0.0", "--port", "8080"]
