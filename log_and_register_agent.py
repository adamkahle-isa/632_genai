"""Log and register AGENT to Unity Catalog with MLflow.

This script follows the Module08 pattern and keeps dependencies minimal.
"""

from pkg_resources import get_distribution

import mlflow

from agent import AGENT


def main() -> None:
    mlflow.set_registry_uri("databricks-uc")
    model_name = "isa632_7474656346303369.fladunhh.genai_project"

    with mlflow.start_run() as run:
        mlflow.models.set_model(AGENT)

        model_info = mlflow.pyfunc.log_model(
            name="agent",
            python_model="agent.py",
            pip_requirements=[
                f"mlflow=={get_distribution('mlflow').version}",
            ],
        )

        model_uri = f"runs:/{run.info.run_id}/{model_info.name}"
        model_version = mlflow.register_model(model_uri, model_name)

        print(f"Run ID: {run.info.run_id}")
        print(f"Model URI: {model_uri}")
        print(f"Registered model: {model_name}")
        print(f"Registered version: {model_version.version}")


if __name__ == "__main__":
    main()
