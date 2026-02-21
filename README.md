# Cats vs Dogs MLOps

This project implements an MLOps pipeline for the Cats vs Dogs image classification problem.

## Structure
- `data/`: Raw and processed data (tracked by DVC)
- `models/`: Model artifacts (tracked by DVC)
- `src/`: Source code (data processing, model, serving, tests)
- `.github/workflows/`: CI/CD workflows
- `docker/`: Docker and Prometheus configs

## Usage
- Place raw data in `data/raw/dogs-vs-cats/`
- Run DVC and pipeline scripts as described in the documentation.
