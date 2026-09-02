terraform {
  backend "s3" {
    bucket         = "sentiment-mlops-tfstate-pdne"
    key            = "classic_ml/terraform.tfstate"
    region         = "eu-west-2"
    dynamodb_table = "sentiment-mlops-tf-lock"
    encrypt        = true
  }
}

provider "aws" {
  region = "eu-west-2"  # London region (picking a region close to you (or your target users) reduces latency.)
}

# ECR repo -> is the storage contains our docker images
resource "aws_ecr_repository" "classic_ml" {
  name = "sentiment-classic"
}

# IAM -> Creates a role and says "Lambda is allowed to become this role"
resource "aws_iam_role" "lambda_exec" {
  name = "sentiment-classic-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

# Attaches an AWS-managed permission set that allows basic things like writing logs to CloudWatch. This is the minimum permission Lambda needs to function at all
resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Lambda function -> pointing at our Docker image
resource "aws_lambda_function" "classic_ml" {
  function_name = "sentiment-classic"
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"   # package_type = "Image" — tells Lambda "you're getting a Docker image"
  image_uri     = "${aws_ecr_repository.classic_ml.repository_url}:latest"  # image_uri — points at the ECR repo you just made, tag latest
  timeout       = 10    # timeout / memory_size — Lambda kills our function if it runs longer than timeout seconds; scikit-learn + our pipeline needs enough memory to load without crashing. 512MB and 10s is a safe starting point for a small model.
  memory_size   = 512
}

# API Gateway -> giving our Lambda a public URL
# define the API Gateway
resource "aws_apigatewayv2_api" "http_api" {
  name          = "sentiment-classic-api"
  protocol_type = "HTTP"
}

# connect API Gateway to lambda
resource "aws_apigatewayv2_integration" "lambda_integration" {
  api_id                 = aws_apigatewayv2_api.http_api.id
  integration_type       = "AWS_PROXY"  # just forward the whole request to Lambda as-is, don't transform it
  integration_uri        = aws_lambda_function.classic_ml.invoke_arn
  payload_format_version = "2.0"
}

# define the URL path for our endpoint
resource "aws_apigatewayv2_route" "predict_route" {
  api_id    = aws_apigatewayv2_api.http_api.id
  route_key = "POST /predict"   # the actual URL path being wired up
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.http_api.id
  name        = "$default"
  auto_deploy = true
}

# Give permission to API Gateway to access this specific Lambda
resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.classic_ml.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http_api.execution_arn}/*/*"
}

# An output so you get the URL after deploying:
output "api_url" {
  value = aws_apigatewayv2_stage.default.invoke_url
}


