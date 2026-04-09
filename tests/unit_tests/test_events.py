from zen_garden.plugin_system.events import Event, Events
from zen_garden.plugin_system.loader import register_plugins


class TestEvents:

    def test_execute_registered_function_when_event_is_triggered(self):
        # Arrange
        spy = []

        @Events.register(Event.test_event)
        def any_function():
            spy.append("any_function is executed")

        # Act
        Events.trigger(Event.test_event)

        # Assert
        assert "any_function is executed" in spy

    def test_pass_arguments_to_observers(self):
        # Arrange
        spy = []

        @Events.register(Event.test_event)
        def any_function(any_argument):
            spy.append(f"{any_argument} has been passed")

        # Act
        Events.trigger(Event.test_event, "any_value")

        # Assert
        assert "any_value has been passed" in spy

    def test_pass_keyword_arguments_to_observers(self):
        # Arrange
        spy = []

        @Events.register(Event.test_event)
        def any_function(any_argument):
            spy.append(f"{any_argument} has been passed")

        # Act
        Events.trigger(Event.test_event, any_argument="any_value")

        # Assert
        assert "any_value has been passed" in spy

    def test_trigger_event_with_no_observer_does_nothing_in_particular(self):
        # Arrange
        spy = []

        @Events.register(Event.test_event)
        def any_function(any_argument):
            spy.append(f"{any_argument} has been passed")

        # Act
        Events.trigger(Event.test_event, any_argument="any_value")

        # Assert
        assert spy == []
