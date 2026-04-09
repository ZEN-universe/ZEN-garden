from zen_garden.plugin_system.events import Event, Events

config = {}


@Events.register(Event.test_event)
def first_method(**kwargs):
    pass