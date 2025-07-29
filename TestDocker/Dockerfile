FROM nvcr.io/nvidia/hpc-benchmarks:24.06

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY code/ /workspace/code/

ENV PYTHONPATH=/workspace