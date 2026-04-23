from zen_garden.plugin_system.events import Event, EventPublisher

config = {}


@EventPublisher.register(Event.test_event1)
def define_global_data(**kwargs):
    """
    Testing method.

    Is registered to test_event1 for testing and stores the value of
    "data_to_keep" in the global variable `stored_data`.
    """
    global stored_data
    stored_data = kwargs["data_to_keep"]


@EventPublisher.register(Event.test_event2)
def append_global_data_to_kwargs(**kwargs):
    """
    Testing method.

    Is registered to test_event2 for testing and appends the value
    of `stored_data` to the list in `kwargs["spy"]`.
    """
    kwargs["spy"].append(stored_data)
