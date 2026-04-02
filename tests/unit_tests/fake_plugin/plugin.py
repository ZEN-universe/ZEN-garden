from zen_garden.events import Events, Event

config = {}

@Events.register(Event.before_optimization_construction)
def first_method(**kwargs):
    global stored_data
    stored_data = kwargs["data_to_keep"]

@Events.register(Event.after_optimization_construction)
def second_method(**kwargs):
    kwargs["spy"].append(stored_data)