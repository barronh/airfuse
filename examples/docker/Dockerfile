FROM python:3.9

# Install necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bash \
    git \
    cron && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
RUN mkdir -p /airfuse/output
RUN mkdir -p /airfuse/logs

# Copy and convert script (ensure no windows line endings)
COPY typical/ /airfuse
RUN find /airfuse -type f -exec sed -i 's/\r$//' {} +
RUN chmod +x /airfuse/run_schedule.sh
RUN chmod +x /airfuse/run.sh

# Install python packages
RUN python -m venv /airfuse/venv && \
    /airfuse/venv/bin/pip install --upgrade pip && \
    /airfuse/venv/bin/pip install -r /airfuse/requirements.txt
    /airfuse/venv/bin/pip install airfuse

# Set environment variables to ensure diskless temp file
ENV TMPDIR=/dev/shm/

# Run the script
CMD ["/bin/bash", "/airfuse/run_schedule.sh"]
