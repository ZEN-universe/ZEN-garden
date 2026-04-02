from zen_garden.events import Event, Events

config = {}
"""
This config dictionary will be filled by 
plugins.loader.register_plugins()
"""

# Choose the event that will trigger the function call
@Events.register(Event.before_optimization_construction)
def this_will_be_called_first(*args, **kwargs):
    """You can name this function as you wish."""
    global data_to_keep
    data_to_keep = kwargs["any_variable"]

# Choose the event that will trigger the function call
@Events.register(Event.before_optimization_construction)
def this_will_be_called_second(*args, **kwargs):
    """You can name this function as you wish."""
    global data_to_keep
    print(f"This was stored before {data_to_keep}")
