FROM continuumio/miniconda3:latest

WORKDIR /workspace

# Install procps so Nextflow can collect task metrics (requires 'ps')
RUN apt-get update && apt-get install -y --no-install-recommends procps && rm -rf /var/lib/apt/lists/*

# Copy environment file
COPY environment-2025-kmerseek-analysis.yml .

# Create conda environment
RUN conda env create -f environment-2025-kmerseek-analysis.yml && \
    conda clean -afy

# Make conda env the default
SHELL ["conda", "run", "-n", "2025-kmerseek-analysis", "/bin/bash", "-c"]

# Copy notebooks
COPY notebooks/ notebooks/
COPY data/ data/

EXPOSE 8888

CMD ["conda", "run", "--no-capture-output", "-n", "2025-kmerseek-analysis", \
     "jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--notebook-dir=/workspace"]
