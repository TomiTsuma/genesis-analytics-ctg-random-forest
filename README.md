## This project entails the workflow for:
1. Exploratory Data Analysis
2. Feature Engineering
3. Feature Selection
4. Model Training
5. Model Evaluation

### pipeline.py contains code to create an airflow Directed Acyclic Graph for model training
This can be used to automate model-training pipeline schedules.
Steps to install

If you are on Windows: Install wsl, Ubuntu and activate Windows Subsystem for Linux setting

Run the following commands:
1. sudo apt-get install software-properties-common
2. sudo apt-get update
3. sudo apt-add-repository universe
4. pip install apache-airflow
5. airflow db init
Move the pipeline folder to /airflow/dags
6. airflow db migrate
7. airflow webserver -p 8000

## The model is served using FastAPI.
! Make sure you move pipeline.py before running this

To run, use this command:
python -m uvicorn app:app --reload 







