# ---- Stage 1: Builder ----
FROM python:3.13-alpine AS builder

WORKDIR /build

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: Runtime ----
FROM python:3.13-alpine

WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

# Copy source code
COPY *.py .

# Switch to non-root user
USER appuser

EXPOSE 3029

CMD ["python", "start.py"]
