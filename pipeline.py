from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
import sys
print(sys.path)
sys.path.append("/mnt/d/Genesis Analytics/app")
import eda, feature_engineering, feature_selection, modeling, outliers, pipeline, visualizations, evaluate
import pandas as pd


default_args = {
    'owner': 'username',
    "email": ["email@domain.com"],
    "email_on_failure": True,
    "email_on_retry": True,
}

base_path = "/mnt/d/Genesis Analytics/app"
conditions = pd.read_csv(f"{base_path}/data/Foetal Health Classifications.csv")
foetal_health = pd.read_csv(f"{base_path}/data/foetal_health.csv")

dag = DAG(
    'ctg_model_training',
    default_args=default_args,
    description = 'Automated Model Training Pipeline',
)

X_train = None
X_valid = None
X_test = None
y_train = None
y_valid = None
y_test = None
y_preds = None
 
config = None

clf = None


def duplicates():
    df = eda.removeDuplicates(foetal_health)
    df.to_csv(f"{base_path}/data/eda_result.csv")
def missingRecords():
    eda.getMissingRecords(foetal_health)
def handleAnomalies():
    eda.getAbnormalData(foetal_health)
def skewNess():
    eda.getSkewness(foetal_health)
def kurtosis():
    eda.getKurtosis(foetal_health)
def dataValidity():
    eda.getDataValidity(foetal_health)
def multicollinearity():
    eda.getMulticollinearity(foetal_health)
def bias():
    eda.getBias(foetal_health)
def chi2Test():
    eda.chiSquareTest(foetal_health)
def ksTest():
    eda.kolmogorovSmirnovTestOfDistribution(foetal_health)
def columnVariance():
    eda.getColumnVariance(foetal_health)

check_duplicates = PythonOperator(
    task_id='check_duplicates',
    # python_callable=fxm,
    python_callable=duplicates,
    dag=dag,
)
check_missing_records = PythonOperator(
    task_id='check_missing_records',
    # python_callable=fxm,
    python_callable=missingRecords,
    dag=dag,
)
handle_anomalies = PythonOperator(
    task_id='handle_anomalies',
    # python_callable=fxm,
    python_callable=handleAnomalies,
    dag=dag,
)
check_skewness = PythonOperator(
    task_id='check_skewness',
    # python_callable=fxm,
    python_callable=skewNess,
    dag=dag,
)
check_kurtosis = PythonOperator(
    task_id='check_kurtosis',
    # python_callable=fxm,
    python_callable=kurtosis,
    dag=dag,
)
check_multicollinearity = PythonOperator(
    task_id='check_multicollinearity',
    # python_callable=fxm,
    python_callable=multicollinearity,
    dag=dag,
)

check_bias = PythonOperator(
    task_id='check_bias',
    # python_callable=fxm,
    python_callable=bias,
    dag=dag,
)
check_chi2_test = PythonOperator(
    task_id='check_chi2_test',
    # python_callable=fxm,
    python_callable=chi2Test,
    dag=dag,
)


def logTrans():
    df = pd.read_csv(f"{base_path}/data/eda_result.csv")
    feature_engineering.logTransformation(df)
def chi2FeatureImportance():
    df = pd.read_csv(f"{base_path}/data/eda_result.csv")
    feature_selection.chiSquareFeatureSelection(df)
def rfImportances():
    df = pd.read_csv(f"{base_path}/data/eda_result.csv")
    feature_selection.rfFeatureImportances(df)
def factAnalysis():
    df = pd.read_csv(f"{base_path}/data/eda_result.csv")
    feature_selection.factorAnalysis(df)
def outlierBoxPlots():
    df = pd.read_csv(f"{base_path}/data/eda_result.csv")
    visualizations.plotOutlierBoxplots(df)
def rstdOutliers():
    df = pd.read_csv(f"{base_path}/data/eda_result.csv")
    outliers.robustStandardDeviationOutliers(df)
def pctOutliers():
    df = pd.read_csv(f"{base_path}/data/eda_result.csv")
    df = outliers.percentileOutliers(df)
def scale():
    df = pd.read_csv(f"{base_path}/data/eda_result.csv")
    feature_engineering.robustScale(df)
def getBiase():
    df = pd.read_csv(f"{base_path}/data/eda_result.csv")
    eda.bias(df)
def binData():
    df = pd.read_csv(f"{base_path}/data/eda_result.csv")
    feature_engineering.quantileBinning(df)
    df.to_csv(f"{base_path}/data/preparation_result.csv")

log_transformation = PythonOperator(
    task_id='log_transformation',
    # python_callable=fxm,
    python_callable=logTrans,
    dag=dag,
)
chi2_feature_importance = PythonOperator(
    task_id='chi2_feature_importance',
    # python_callable=fxm,
    python_callable=chi2FeatureImportance,
    dag=dag,
)
rf_mportances = PythonOperator(
    task_id='rf_mportances',
    # python_callable=fxm,
    python_callable=rfImportances,
    dag=dag,
)
factor_analysis = PythonOperator(
    task_id='factor_analysis',
    # python_callable=fxm,
    python_callable=factAnalysis,
    dag=dag,
)
outlier_box_plots = PythonOperator(
    task_id='outlier_box_plots',
    # python_callable=fxm,
    python_callable=outlierBoxPlots,
    dag=dag,
)
check_chi2_test = PythonOperator(
    task_id='check_chi2_test',
    # python_callable=fxm,
    python_callable=chi2Test,
    dag=dag,
)
rstd_outliers = PythonOperator(
    task_id='robust_standard_deviation_outliers',
    # python_callable=fxm,
    python_callable=rstdOutliers,
    dag=dag,
)
pct_outliers = PythonOperator(
    task_id='percentile_outliers',
    # python_callable=fxm,
    python_callable=pctOutliers,
    dag=dag,
)
robust_scaling = PythonOperator(
    task_id='robust_scaling',
    # python_callable=fxm,
    python_callable=scale,
    dag=dag,
)
check_bias2 = PythonOperator(
    task_id='check_bias_after_scaling',
    # python_callable=fxm,
    python_callable=getBiase,
    dag=dag,
)
bin_data = PythonOperator(
    task_id='bin_data',
    # python_callable=fxm,
    python_callable=binData,
    dag=dag,
)
def split():
    df = pd.read_csv(f"{base_path}/data/preparation_result.csv")
    global X_train, X_valid, X_test, y_train, y_valid, y_test
    X_train, X_valid, X_test, y_train, y_valid, y_test = modeling.splitData(df.drop("fetal_health",axis=1), df['fetal_health'])

def hyperparams():
    global X_valid, X_test, y_valid, y_test
    config = modeling.hyperparamTuning(X_valid, X_test, y_valid, y_test)

def oversample():
    global X_train, y_train
    X_train, y_train = modeling.overSample(X_train, y_train)

def train():
    global clf, config, X_train, y_train
    clf = modeling.trainRFClassifier(config, X_train, y_train)

def evaluate():
    global  y_tes, y_preds
    evaluate.confusionMatrix(y_test, y_preds, columns =y_test.unique()[::-1])
    evaluate.aucROC(y_preds, y_test)

split_data = PythonOperator(
    task_id='split_data',
    # python_callable=fxm,
    python_callable=split,
    dag=dag,
)
hyperparams_tuning = PythonOperator(
    task_id='hyperparams_tuning',
    # python_callable=fxm,
    python_callable=hyperparams,
    dag=dag,
)
smote_oversampling_data = PythonOperator(
    task_id='smote_oversampling_data',
    # python_callable=fxm,
    python_callable=oversample,
    dag=dag,
)
train_rf_model = PythonOperator(
    task_id='train_rf_model',
    # python_callable=fxm,
    python_callable=train,
    dag=dag,
)
evaluate_model = PythonOperator(
    task_id='evaluate_model',
    # python_callable=fxm,
    python_callable=evaluate,
    dag=dag,
)

end_task = DummyOperator(task_id='end_task', dag=dag)

check_duplicates >> check_missing_records >> handle_anomalies >> check_skewness >> check_kurtosis >> check_multicollinearity >> check_bias >> check_chi2_test

check_chi2_test >> log_transformation >> chi2_feature_importance >> rf_mportances >> factor_analysis >> outlier_box_plots >> check_chi2_test >> rstd_outliers >> pct_outliers >> robust_scaling >> check_bias2 >> bin_data

bin_data >> split_data >> hyperparams_tuning >> smote_oversampling_data >> train_rf_model >> evaluate_model

evaluate_model >> end_task
