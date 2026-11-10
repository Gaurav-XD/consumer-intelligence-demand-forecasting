FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY src /app/src
COPY models/churn/tuned_best_model.joblib /app/models/churn/tuned_best_model.joblib
COPY models/forecasting/tuned_best_model.joblib /app/models/forecasting/tuned_best_model.joblib
COPY reports/churn_tuning_summary.json /app/reports/churn_tuning_summary.json
COPY reports/forecasting_tuning_summary.json /app/reports/forecasting_tuning_summary.json

RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "consumer_intelligence.api:app", "--host", "0.0.0.0", "--port", "8000"]
