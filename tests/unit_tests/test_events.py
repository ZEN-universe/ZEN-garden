"""Unit tests for the event publishing system.

Unit tests for `EventPublisher` and `Event` semantics.
"""

from enum import auto

from zen_garden.plugin_system.events import Event, EventPublisher


class TestEvent(Event):
    """Enum of test events used by the tests.

    Attributes:
        test_event1: First test event.
        test_event2: Second test event.
    """

    test_event1 = auto()
    test_event2 = auto()


class TestEvents:
    """Tests for event registration and notification.

    Each method verifies one aspect of the `EventPublisher` API.
    """

    def test_execute_registered_function_when_event_is_triggered(self):
        """Registered observers are called when their event is triggered.

        Verifies that a function registered to an event is executed on trigger.
        """
        spy = []

        @EventPublisher.register(TestEvent.test_event1)
        def any_function():
            spy.append("any_function is executed")

        # Act
        EventPublisher.trigger(TestEvent.test_event1)

        # Assert
        assert "any_function is executed" in spy

    def test_pass_arguments_to_observers(self):
        """Positional arguments are forwarded to observers.

        The observer should receive positional arguments passed to trigger.
        """
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
        """Keyword arguments are forwarded to observers.

        The observer should receive keyword arguments passed to trigger.
        """
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
        """Triggering an unobserved event has no side effects.

        Confirms that triggering another event without observers leaves state unchanged.
        """
        # Arrange
        spy = []

        @EventPublisher.register(TestEvent.test_event1)
        def any_function(any_argument):
            spy.append(f"{any_argument} has been passed")

        # Act
        EventPublisher.trigger(TestEvent.test_event2, any_argument="any_value")

        # Assert
        assert spy == []
