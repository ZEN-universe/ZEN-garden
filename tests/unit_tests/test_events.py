from zen_garden.plugin_system.events import EventPublisher, Event
from enum import auto

class TestEvent(Event):
    test_event1 = auto()
    test_event2 = auto()

class TestEvents:

    def test_execute_registered_function_when_event_is_triggered(self):
        # Arrange
        spy = []

        @EventPublisher.register(TestEvent.test_event1)
        def any_function():
            spy.append("any_function is executed")

        # Act
        EventPublisher.trigger(TestEvent.test_event1)

        # Assert
        assert "any_function is executed" in spy

    def test_pass_arguments_to_observers(self):
        # Arrange
        spy = []

        @EventPublisher.register(TestEvent.test_event1)
        def any_function(any_argument):
            spy.append(f"{any_argument} has been passed")

        # Act
        EventPublisher.trigger(TestEvent.test_event1, "any_value")

        # Assert
        assert "any_value has been passed" in spy

    def test_pass_keyword_arguments_to_observers(self):
        # Arrange
        spy = []

        @EventPublisher.register(TestEvent.test_event1)
        def any_function(any_argument):
            spy.append(f"{any_argument} has been passed")

        # Act
        EventPublisher.trigger(TestEvent.test_event1, any_argument="any_value")

        # Assert
        assert "any_value has been passed" in spy

    def test_trigger_event_with_no_observer_does_nothing_in_particular(self):
        # Arrange
        spy = []

        @EventPublisher.register(TestEvent.test_event1)
        def any_function(any_argument):
            spy.append(f"{any_argument} has been passed")

        # Act
        EventPublisher.trigger(TestEvent.test_event2, any_argument="any_value")

        # Assert
        assert spy == []
