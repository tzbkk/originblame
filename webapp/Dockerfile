FROM node:22-slim AS frontend-builder

WORKDIR /build
COPY originblame/webapp/frontend/package.json originblame/webapp/frontend/package-lock.json ./
RUN npm ci
COPY originblame/webapp/frontend/ ./
RUN npm run build

FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends nginx && \
    rm -rf /var/lib/apt/lists/* && \
    rm -f /etc/nginx/sites-enabled/default

COPY originblame/webapp/backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY rust-originblame/python/src/ob/ /app/ob/
ENV OB_PYTHON_SRC=/app
ENV PYTHONPATH=/app

COPY originblame/webapp/backend/ /app/backend/
COPY --from=frontend-builder /build/dist /app/frontend
COPY originblame/webapp/docker/nginx.conf /etc/nginx/sites-available/originblame
RUN ln -sf /etc/nginx/sites-available/originblame /etc/nginx/sites-enabled/originblame

ENV OB_PIPELINE_DIR=/data/pipeline_v2
ENV OB_QA_DIR=/data/pipeline_qa

COPY originblame/benchmarks/results/pipeline_v2/huggingface-zhwiki-1k-ob /data/pipeline_v2/huggingface-zhwiki-1k-ob
COPY originblame/benchmarks/results/pipeline_v2/huggingface-zhwiki-10k-ob /data/pipeline_v2/huggingface-zhwiki-10k-ob
COPY originblame/benchmarks/results/pipeline_v2/huggingface-zhwiki-100k-ob /data/pipeline_v2/huggingface-zhwiki-100k-ob
COPY originblame/benchmarks/results/pipeline_v2/huggingface-zhwiki-all-ob /data/pipeline_v2/huggingface-zhwiki-all-ob
COPY originblame/benchmarks/results/pipeline_v2/huggingface-kernel-1k-ob /data/pipeline_v2/huggingface-kernel-1k-ob
COPY originblame/benchmarks/results/pipeline_v2/huggingface-kernel-10k-ob /data/pipeline_v2/huggingface-kernel-10k-ob
COPY originblame/benchmarks/results/pipeline_v2/huggingface-kernel-all-ob /data/pipeline_v2/huggingface-kernel-all-ob
COPY originblame/benchmarks/results/pipeline_qa/qa_chatml /data/pipeline_qa/qa_chatml

COPY originblame/webapp/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 80
ENTRYPOINT ["/entrypoint.sh"]
