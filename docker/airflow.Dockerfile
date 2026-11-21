FROM apache/airflow:2.9.3-python3.10
COPY requirements-ml.txt /tmp/requirements-ml.txt
RUN pip install --no-cache-dir -r /tmp/requirements-ml.txt
