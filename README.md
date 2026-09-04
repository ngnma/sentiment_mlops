# Sentiment Classifier — MLOps Deployment on AWS

A production-style deployment of a machine learning sentiment classifier, built to demonstrate
end-to-end MLOps practices: experiment tracking, containerization, infrastructure as code,
CI/CD, and observability on AWS.

**Live stack:** scikit-learn → MLflow → FastAPI → Docker → AWS Lambda → API Gateway → CloudWatch,
provisioned entirely through Terraform and deployed via GitHub Actions.

> This repository currently implements the classic ML (scikit-learn) serving path. A second,
> transformer-based model served via Amazon SageMaker is planned as a parallel deployment path.

---

## Highlights

- **Infrastructure as Code** — 100% of AWS infrastructure (compute, networking, IAM, monitoring)
  defined in Terraform, with remote state management (S3 + DynamoDB locking) for safe collaboration
  between local development and CI/CD.
- **CI/CD pipeline** — GitHub Actions automatically tests, builds, and deploys on every push:
  no manual deployment steps.
- **Experiment tracking & model registry** — MLflow tracks every training run's parameters and
  metrics, with the best model versioned in the MLflow Model Registry.
- **Serverless, cost-efficient architecture** — AWS Lambda + API Gateway scale to zero; no idle
  compute cost.
- **Observability** — CloudWatch metrics, logs, and automated alarm-based alerting (via SNS) on
  error thresholds.
- **Containerized, cloud-portable service** — a standard FastAPI application, decoupled from the
  Lambda runtime via an adapter layer, so the same service could be redeployed on ECS, EKS, or any
  container platform with minimal changes.

---

## Architecture

```
┌────────────────┐     ┌─────────────┐     ┌──────────────────────────┐
│  Training       │────▶│   MLflow    │────▶│  Model Registry           │
│  (scikit-learn) │     │  Tracking   │     │  (sentiment-classic-v1)   │
└────────────────┘     └─────────────┘     └──────────────────────────┘
                                                        │
                                                        ▼
                                          ┌──────────────────────────┐
                                          │  FastAPI + Mangum         │
                                          │  (containerized)          │
                                          └──────────────────────────┘
                                                        │
                                                        ▼  Docker image → Amazon ECR
                                          ┌──────────────────────────┐
                                          │  AWS Lambda                │
                                          │  (container image runtime) │
                                          └──────────────────────────┘
                                                        │
                                                        ▼
                                          ┌──────────────────────────┐
                                          │  API Gateway (HTTP API)    │
                                          │  POST /predict              │
                                          └──────────────────────────┘
                                                        │
                                                        ▼
                                          ┌──────────────────────────┐
                                          │  CloudWatch                │
                                          │  Logs · Metrics · Alarms   │
                                          │  → SNS email alerts        │
                                          └──────────────────────────┘

Infrastructure provisioned via Terraform | State: S3 (remote) + DynamoDB (locking)
Deployment automated via GitHub Actions (test → build → push → apply)
```

---

## Tech Stack

| Category | Technology |
|---|---|
| Modeling | Python, scikit-learn, GridSearchCV |
| Experiment tracking / Model registry | MLflow |
| API framework | FastAPI |
| Serverless adapter | Mangum |
| Containerization | Docker |
| Compute | AWS Lambda (container image) |
| API layer | Amazon API Gateway (HTTP API) |
| Container registry | Amazon ECR |
| Infrastructure as Code | Terraform (S3 + DynamoDB remote backend) |
| CI/CD | GitHub Actions |
| Observability | Amazon CloudWatch, Amazon SNS |
| Identity & access | AWS IAM (least-privilege roles) |

---

## Repository Structure

```
classic_ml/           Training script, FastAPI service, Dockerfile, unit tests
infra/
  bootstrap/           Terraform: remote state backend (S3 + DynamoDB)
  classic_ml/          Terraform: ECR, IAM, Lambda, API Gateway, CloudWatch alarm
model/                  Serialized model artifact
notebooks/              Exploratory data analysis and model development
.github/workflows/      CI/CD pipeline definitions
requirements.txt
run_mlflow.sh
```

---

## System Design Notes

**Serving layer.** The model is served through a standard FastAPI application rather than
Lambda-specific code, keeping the service portable across compute targets. [Mangum](https://github.com/jordaneremieff/mangum)
adapts FastAPI's ASGI interface to AWS Lambda's event-driven invocation model, so the same
codebase runs unmodified in a local dev server or inside Lambda.

**Infrastructure.** All resources — ECR repository, IAM execution role, Lambda function, API
Gateway route/integration, and CloudWatch alarm — are declared in Terraform. Terraform state is
stored remotely in S3 with DynamoDB-backed locking, so the CI/CD pipeline and local development
operate against a single source of truth for infrastructure state, avoiding drift or duplicate
resource creation.

**CI/CD.** A GitHub Actions workflow triggers on pushes to the relevant paths, running the test
suite, rebuilding and pushing the container image to ECR, and applying the Terraform
configuration — making every deployment reproducible and auditable via the Git history.

**Observability.** CloudWatch captures structured invocation logs and standard Lambda metrics
(invocation count, error rate, duration) with no additional instrumentation required. A
CloudWatch alarm monitors the error rate and publishes to an SNS topic for email alerting.
Prediction inputs/outputs are additionally logged to support future data drift analysis.

**Cost model.** The architecture is intentionally serverless end-to-end: Lambda and API Gateway
incur no idle cost and bill only per invocation, making the service cost-efficient at any traffic
level, including zero.

---

## Getting Started

### Prerequisites
- Python 3.12, Docker, Terraform, AWS CLI (configured with an IAM user)

### Local development

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

./run_mlflow.sh                        # start local MLflow tracking server
python classic_ml/train.py             # train + log to MLflow

uvicorn classic_ml.app:app --reload --port 8000
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" -d '{"text": "I love this product"}'
```

### Deployment

```bash
# one-time: provision the Terraform remote state backend
cd infra/bootstrap && terraform init && terraform apply

# build and push the container image
docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
  -t sentiment-classic -f classic_ml/Dockerfile . --load
docker push <account-id>.dkr.ecr.eu-west-2.amazonaws.com/sentiment-classic:latest

# provision application infrastructure
cd infra/classic_ml && terraform init && terraform apply
```

Subsequent pushes to `main` affecting `classic_ml/` or `infra/classic_ml/` trigger the CI/CD
pipeline automatically.

---

## API

**POST `/predict`**

```json
// Request
{ "text": "I love this product" }

// Response
{ "label": "positive", "score": 0.69 }
```

**GET `/health`** — liveness check, returns `{"status": "ok"}`.

---

## Roadmap

- [ ] Load model dynamically from MLflow Model Registry at runtime rather than baking into the image
- [ ] Sampled prediction logging to S3 with a scheduled drift-analysis job
- [ ] Second deployment path: transformer model (HuggingFace/PyTorch) via SageMaker Serverless Inference
- [ ] Cross-model benchmarking (latency, cost, accuracy) between the two serving paths

---

## License

MIT