from zen_garden.events import Event, Events

config = {}


@Events.register(Event.test_event)
def first_method(**kwargs):
    pass