# start from AWS's pre-built base
FROM public.ecr.aws/lambda/python:3.12 

# copy and install dependencies inside the image
COPY requirements.txt .
RUN pip install -r requirements.txt

# copy app code inside the image
COPY classic_ml/app.py ${LAMBDA_TASK_ROOT}/app.py
# copy trained model inside the image
COPY model/classic_pipeline.joblib ${LAMBDA_TASK_ROOT}/model/classic_pipeline.joblib

# tell Lambda what to run when started: "app.handler" = handler variable inside app.py
CMD ["app.handler"]