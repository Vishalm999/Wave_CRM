# Use an official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /new_source_funnel

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8080

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "new_source_funnel:app", "--host", "0.0.0.0", "--port", "8080"]
