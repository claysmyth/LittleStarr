import ray, os, hydra, yaml
from sklearn.tree import DecisionTreeClassifier
from utils import setup_ray
from model_development.embedded_model_classes import TwoStepLDATree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from powerband_identification import powerband_identification_pipeline

DB_PATH = "/media/shortterm_ssd/Clay/databases/duckdb/rcs-db.duckdb"


# The rationale for 24 cpus: Max batch is 4 devices, and each device will use 5 cores for the cross-validation steps.
# Allocating 1 extra process for cross-validation allows for 6 cores per device.
# According to the documentation, num_cpus provides a 'max number' of cpu cores for that remote task.
# Reduced to 12 because I think I overworked the server and it automatically restarted due to overheating or overload lol
@ray.remote(num_cpus=20)
def hyperparameter_search_wrapper(
    devices,
    params,
    sleep_stage_mapping,
    file_name,
    BASE_PATH,
):
    for device in devices:
        out_path = BASE_PATH + file_name
        powerband_identification_pipeline(
            device,
            params,
            sleep_stage_mapping,
            out_file_path=out_path,
            db_path=DB_PATH,
        )


def run_parallelized_pipeline(
    parameters,
    sleep_stage_mapping,
    batches,
    file_name,
    BASE_PATH,
):
    """
    Run the pipeline in parallel using Ray. IMPORTANT TO SET PARAMETERS!
    """
    full_batch = [
        hyperparameter_search_wrapper.remote(
            batch, parameters, sleep_stage_mapping, file_name, BASE_PATH
        )
        for batch in batches
    ]
    execute = ray.get(full_batch)


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    sleep_stage_mapping = cfg.sleep_stage_mapping
    # batches = cfg.batches
    file_name = cfg.file_name
    BASE_PATH = cfg.BASE_PATH
    use_ray = cfg.use_ray
    device = cfg.device
    session_validation = cfg.leave_one_session_out_validation

    # Create output directory
    Path(BASE_PATH).mkdir(parents=True, exist_ok=True)

    # Convert parameters from OmegaConf object to dict
    parameters = OmegaConf.to_container(
        cfg.parameters, resolve=True
    ) | OmegaConf.to_container(cfg.method_params.parameters, resolve=True)
    parameters["method"] = cfg.method_params.method

    if parameters["method"] == "sfs_cluster":
        parameters = parameters | OmegaConf.to_container(
            cfg.method_params.feature_filter, resolve=True
        )

    match parameters["model"]:
        case "LinearDiscriminantAnalysis":
            parameters["model"] = LinearDiscriminantAnalysis()
        case "DecisionTreeClassifier":
            parameters["model"] = DecisionTreeClassifier(max_leaf_nodes=3)
        case "TwoStepLDATree":
            parameters["model"] = TwoStepLDATree()
        case _:
            raise ValueError(f'Invalid model type: {parameters["model"]}')

    # Save parameters to yaml
    with open(
        f"{BASE_PATH}{os.path.splitext(file_name)[0]}_paramaters.yaml",
        "w",
        encoding="utf-8",
    ) as file:
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), file)

    if use_ray:
        setup_ray(allocate_memory=False)
        run_parallelized_pipeline(
            parameters, sleep_stage_mapping, batches, file_name, BASE_PATH=BASE_PATH
        )
        ray.shutdown()
    else:
        powerband_identification_pipeline(
            device,
            parameters,
            sleep_stage_mapping,
            out_file_path=BASE_PATH + file_name,
            db_path=DB_PATH,
            session_validation=session_validation,
        )


if __name__ == "__main__":
    main()
