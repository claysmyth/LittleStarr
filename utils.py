import ray
import json

def setup_ray(
    num_GB_to_allocate=50,
    allocate_memory=False,
    base_path="/media/shortterm_ssd/Clay/ray_spill/",
):
    memory_allocation_in_bytes = num_GB_to_allocate * 1024**3
    # The object_store_memory flag tells ray how much memory to allocate to the object store (stored in RAM, as a RAM disk. The object store is written to tmpfs, mounted on /dev/shm),
    # before spill over to disk occurs. One should consider using the allocate_memory flag if RAM is tight and spillover to disk is desired and controlled.
    if allocate_memory:
        ray.init(
            object_store_memory=memory_allocation_in_bytes,
            _system_config={
                "max_io_workers": 4,  # More IO workers for parallelism.
                "object_spilling_config": json.dumps(
                    {
                        "type": "filesystem",
                        "params": {
                            # Multiple directories can be specified to distribute
                            # IO across multiple mounted physical devices.
                            "directory_path": [
                                base_path + "/tmp/spill",
                                base_path + "/tmp/spill_1",
                                base_path + "/tmp/spill_2",
                            ]
                        },
                    }
                ),
            },
        )
    else:
        # I don't think spill-over is working correctly with current config... Raylet was killing jobs because of 'out of memory' errors.
        ray.init(
            _system_config={
                "max_io_workers": 4,  # More IO workers for parallelism.
                "object_spilling_config": json.dumps(
                    {
                        "type": "filesystem",
                        "params": {
                            # Multiple directories can be specified to distribute
                            # IO across multiple mounted physical devices.
                            "directory_path": [
                                base_path + "/tmp/spill",
                                base_path + "/tmp/spill_1",
                                base_path + "/tmp/spill_2",
                            ]
                        },
                    }
                ),
            }
        )