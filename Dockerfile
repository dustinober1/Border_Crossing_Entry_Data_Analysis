# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create directories for outputs
RUN mkdir -p results/visualizations

# Expose ports for different services
EXPOSE 8050 8501

# Create startup script
RUN echo '#!/bin/bash\n\
echo "🌍 Border Crossing Analytics - Container Starting"\n\
echo "Available services:"\n\
echo "  - Interactive Dashboard: http://localhost:8050"\n\
echo "  - Portfolio Website: http://localhost:8501"\n\
echo "  - ML Analysis: python advanced_ml_analysis.py"\n\
echo "  - Anomaly Detection: python anomaly_detection_system.py"\n\
echo "  - Report Generation: python automated_report_generator.py"\n\
echo ""\n\
echo "Starting services..."\n\
python interactive_dashboard.py &\n\
streamlit run portfolio_website.py\n\
' > start.sh && chmod +x start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["./start.sh"]

# Labels for documentation
LABEL maintainer="Border Crossing Analytics Team"
LABEL version="2.0"
LABEL description="Advanced Border Crossing Analytics Platform with ML, Dashboards, and Automated Reporting"