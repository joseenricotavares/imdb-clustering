FROM python:3.11-slim

WORKDIR /app

# Copies only the essential files
COPY app.py .
COPY requirements.txt .
COPY data/* data/
COPY models/* models/

# Upgrades pip and installs dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Exposes Streamlit port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py"]