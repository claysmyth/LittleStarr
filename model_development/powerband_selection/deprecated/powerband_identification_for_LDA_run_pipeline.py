
import ray
import json
import sys
from sklearn.tree import DecisionTreeClassifier
# sys.path.append('powerband_identification_for_LDA_pipeline_funcs.py')
# from powerband_identification_for_LDA_pipeline_funcs import hyperparameter_search_pipeline

#sys.path.append('/model_development/powerband_identification_for_LDA_pipeline_funcs_no_feature_eng.py')
from powerband_identification_for_LDA_no_feature_eng import hyperparameter_search_pipeline
#sys.path.append('embedded_model_classes.py')
from embedded_model_classes import TwoStepLDATree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import lightgbm as lgb
import os
from pathlib import Path
import hydra
from omegaconf import OmegaConf, DictConfig
import yaml

def setup_ray(num_GB_to_allocate = 50, allocate_memory=False, base_path='/media/shortterm_ssd/Clay/ray_spill/'):
    memory_allocation_in_bytes = num_GB_to_allocate * 1024**3
    # The object_store_memory flag tells ray how much memory to allocate to the object store (stored in RAM, as a RAM disk. The object store is written to tmpfs, mounted on /dev/shm),
    # before spill over to disk occurs. One should consider using the allocate_memory flag if RAM is tight and spillover to disk is desired and controlled.
    if allocate_memory:
        ray.init(object_store_memory=memory_allocation_in_bytes,
                _system_config={
                                "max_io_workers": 4,  # More IO workers for parallelism.
                                    "object_spilling_config": json.dumps(
                                        {
                                            "type": "filesystem",
                                            "params": {
                                                # Multiple directories can be specified to distribute
                                                # IO across multiple mounted physical devices.
                                                "directory_path": [
                                                base_path+"/tmp/spill",
                                                base_path+"/tmp/spill_1",
                                                base_path+"/tmp/spill_2",
                                                ]
                                            },
                                        }
                                )
                })
    else:
        # I don't think spill-over is working correctly with current config... Raylet was killing jobs because of 'out of memory' errors.
        ray.init(_system_config={
                                "max_io_workers": 4,  # More IO workers for parallelism.
                                "object_spilling_config": json.dumps(
                                        {
                                            "type": "filesystem",
                                            "params": {
                                                # Multiple directories can be specified to distribute
                                                # IO across multiple mounted physical devices.
                                                "directory_path": [
                                                base_path+"/tmp/spill",
                                                base_path+"/tmp/spill_1",
                                                base_path+"/tmp/spill_2",
                                                ]
                                            },
                                        }
                                )
                })





def run_parallelized_pipeline(parameters, sleep_stage_mapping, batches, file_name, BASE_PATH='/media/shortterm_ssd/Clay/databases/Sleep_10day_with_autonomic/'):
    """
    Run the pipeline in parallel using Ray. IMPORTANT TO SET PARAMETERS!
    """
    # The rationale for 24 cpus: Max batch is 4 devices, and each device will use 5 cores for the cross-validation steps. 
    # Allocating 1 extra process for cross-validation allows for 6 cores per device.
    # According to the documentation, num_cpus provides a 'max number' of cpu cores for that remote task.

    # Reduced to 12 because I think I overworked the server and it automatically restarted due to overheating or overload lol
    @ray.remote(num_cpus=20) 
    def hyperparameter_search_wrapper(devices, params, sleep_stage_mapping, file_name, BASE_PATH='/media/shortterm_ssd/Clay/databases/Sleep_10day_with_autonomic/'):
        for device in devices:
            DEVICE_PATH =  BASE_PATH + f'RCS{device}/'
            Path(DEVICE_PATH).mkdir(parents=False, exist_ok=True)
            out_path = DEVICE_PATH + file_name
            #hyperparameter_search_pipeline(device, params, sleep_stage_mapping, out_file_path=out_path, use_LDA=False)
            hyperparameter_search_pipeline(device, params, sleep_stage_mapping, out_file_path=out_path, db_path='/media/shortterm_ssd/Clay/databases/duckdb/rcs-db.duckdb')
        
    full_batch = [hyperparameter_search_wrapper.remote(batch, parameters, sleep_stage_mapping, file_name, BASE_PATH) for batch in batches]
    execute = ray.get(full_batch)

@hydra.main(version_base=None, config_path='./conf', config_name='pb_identification_config')
def main(cfg: DictConfig):
    # Use Ray launcher?
    
    # Parameters to be used in the pipeline
    parameters = cfg.parameters
    sleep_stage_mapping = cfg.sleep_stage_mapping
    batches = cfg.batches
    file_name = cfg.file_name
    BASE_PATH = cfg.BASE_PATH

    # Convert parameters from OmegaConf object to dict
    parameters = OmegaConf.to_container(parameters, resolve=True)
    match parameters['model']:
        case 'LinearDiscriminantAnalysis':
            parameters['model'] = LinearDiscriminantAnalysis()
        case 'DecisionTreeClassifier':
            parameters['model'] = DecisionTreeClassifier()
        case 'TwoStepLDATree':
            parameters['model'] = TwoStepLDATree()
        case _:
            raise ValueError(f'Invalid model type: {parameters["model"]}')

    # Verify parameters
    proceed = input(f'Run pipeline with: \n parameters: {parameters} \n sleep stage mapping: {sleep_stage_mapping}' + 
                    f'\n device batching: {batches} \n output file name: {file_name} \n ? (y/n)')

    # Save parameters to json
    # TODO: Convert to yaml
    # with open(f'{BASE_PATH}{os.path.splitext(file_name)[0]}_paramaters.json', 'w', encoding="utf-8") as file:
    #     json.dump({**{key: (str(value) if ~isinstance(value, str) else value) for key, value in parameters.items()}, 
    #                'sleep_stage_mapping': sleep_stage_mapping, 'batches': batches}, file)

    # Save parameters to yaml    
    with open(f'{BASE_PATH}{os.path.splitext(file_name)[0]}_paramaters.yaml', 'w', encoding="utf-8") as file:
        yaml.dump(OmegaConf.to_yaml(cfg), file)

    if proceed == 'y' or proceed == 'Y':
        setup_ray(allocate_memory=False)
        run_parallelized_pipeline(parameters, sleep_stage_mapping, batches, file_name, BASE_PATH=BASE_PATH)
    else:
        print('Pipeline not run.')
    
    ray.shutdown()


if __name__ == '__main__':
    main()