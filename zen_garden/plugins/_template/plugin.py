"""
A template for a plugin

Write functions that subscribe to an event if decorated.These functions are executed
when the execution reaches the trigger to the respective event.
"""

from zen_garden.plugin_system.events import Event, Events

config = {}
"""
This config dictionary will be filled by 
plugins.loader.register_plugins()
"""

# Choose the event that will trigger the function call
@Events.register(Event.test_event)
def this_will_be_called_first(*args, **kwargs):
    """You can name this function as you wish."""
    pass
