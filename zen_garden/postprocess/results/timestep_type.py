from enum import Enum


class TimestepType(Enum):
    yearly = "year"
    operational = "time_operation"
    storage = "time_storage_level"

    @classmethod
    def get_names(cls) -> list[str]:
        """Get a list of timestep names."""
        return [time_step_type.value for time_step_type in cls]

    @classmethod
    def from_index_names(
        cls, index_names: list[str]
    ) -> "tuple[str, TimestepType] | tuple[None, None]":
        """Get the timestep type given a timestep name.
        :param time_step: The name of the timestep.
        :return: The timestep type.
        """
        TIME_INDEX_MAP = {
            "set_years": TimestepType.yearly,
            "set_time_steps_operation": TimestepType.operational,
            "set_time_steps_storage_level": TimestepType.storage,
            "set_time_steps_storage": TimestepType.storage,
        }
        time_index = set(index_names).intersection(set(TIME_INDEX_MAP.keys()))
        timestep_name = time_index.pop() if len(time_index) > 0 else None

        if timestep_name is None:
            return None, None
        return timestep_name, TIME_INDEX_MAP[timestep_name]
