import mlflow


def setup_experiment():
    mlflow.set_experiment("card-fraud-detection")
    print("MLflow experiment ready!")

def log_metrics(metrics):
    mlflow.log_metrics(metrics)

def log_params(params):
    mlflow.log_params(params)

def log_model_artifact(model_path):
    mlflow.log_artifact(model_path)

def track_experiment(run_name, params, metrics, model_path):

    with mlflow.start_run(run_name=run_name):
        log_params(params)
        log_metrics(metrics)
        log_model_artifact(model_path)

    print(f"Experiment '{run_name}' logged successfully!")