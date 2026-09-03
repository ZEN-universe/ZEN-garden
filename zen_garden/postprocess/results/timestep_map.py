from dataclasses import dataclass


@dataclass
class TimestepMap:
    operation: list[int]
    storage: list[int]
    yearly: list[int]
    optimized_time_steps: list[int]
    time_steps_operation_duration: dict[str, str]
    time_steps_storage_duration: dict[str, str]
    time_steps_storage_level_startend_year: dict[str, str]
    time_steps_year2operation: dict[str, list[int]]
    time_steps_year2storage: dict[str, list[int]]
