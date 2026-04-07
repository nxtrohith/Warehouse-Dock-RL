---
title: Warehouse Dock OpenEnv
emoji: 🚚
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Warehouse Dock OpenEnv

This project exposes a simple HTTP API for the warehouse dock environment.

Mandatory endpoints:

- `POST /reset`
- `POST /step`

## Local Docker

Build image:

```bash
docker build . -t warehouse-openenv:latest
```

Run container:

```bash
docker run --rm -p 7860:7860 warehouse-openenv:latest
```

## API Usage

Reset environment:

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 7, "max_steps": 32}'
```

Step environment:

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": 0}'
```

## Hugging Face Space Deployment

1. Create a new Hugging Face Space and choose **Docker**.
2. Push this repository (including `Dockerfile` and this `README.md`).
3. Space will build automatically and serve the app on port `7860`.
