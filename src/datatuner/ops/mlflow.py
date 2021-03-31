import mlflow


def get_artifact(run_id, path):
    client = mlflow.tracking.MlflowClient()
    return client.download_artifacts(run_id, path)


def get_finished_models(experiments):
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiments, filter_string="metrics.finished=1")
    run_ids = [x.info.run_id for x in runs]
    return run_ids
