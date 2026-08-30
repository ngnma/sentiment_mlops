FROM public.ecr.aws/lambda/python:3.12

# copy dependency list and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# copy your app code and the trained model
COPY classic_ml/app.py ${LAMBDA_TASK_ROOT}/app.py
COPY model/classic_pipeline.joblib ${LAMBDA_TASK_ROOT}/model/classic_pipeline.joblib

# tell Lambda which function to call: "app.handler" = handler variable inside app.py
CMD ["app.handler"]