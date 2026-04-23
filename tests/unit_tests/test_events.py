"""Unit tests for the event publishing system.

Unit tests for `EventPublisher` and `Event` semantics.
"""

import pytest

from zen_garden.plugin_system.events import Event, EventPublisher
from zen_garden.plugin_system.loader import register_plugins


@pytest.fixture(scope="function", autouse=True)
def cleanup(request: pytest.FixtureRequest):
    """
    Pytest fixture to clean up registered observers after each test.
    """
    request.addfinalizer(EventPublisher.deregister_all)


class TestEvents:
    """Tests for event registration and notification.

    Each method verifies one aspect of the `EventPublisher` API.
    """

    def test_execute_registered_function_when_event_is_triggered(self):
        """Registered observers are called when their event is triggered.

        Verifies that a function registered to an event is executed on trigger.
        """
        spy = []

        @EventPublisher.register(Event.test_event1)
        def any_function():
            spy.append("any_function is executed")

        # Act
        EventPublisher.trigger(Event.test_event1)

        # Assert
        assert "any_function is executed" in spy

    def test_pass_arguments_to_observers(self):
        """Positional arguments are forwarded to observers.

        The observer should receive positional arguments passed to trigger.
        """
        # Arrange
        spy = []

        @EventPublisher.register(Event.test_event1)
        def any_function(any_argument):
            spy.append(f"{any_argument} has been passed")

        # Act
        EventPublisher.trigger(Event.test_event1, "any_value")

        # Assert
        assert "any_value has been passed" in spy

    def test_pass_keyword_arguments_to_observers(self):
        """Keyword arguments are forwarded to observers.

        The observer should receive keyword arguments passed to trigger.
        """
        # Arrange
        spy = []

        @EventPublisher.register(Event.test_event1)
        def any_function(any_argument):
            spy.append(f"{any_argument} has been passed")

        # Act
        EventPublisher.trigger(Event.test_event1, any_argument="any_value")

        # Assert
        assert "any_value has been passed" in spy

    def test_trigger_event_with_no_observer_does_nothing_in_particular(self):
        """Triggering an unobserved event has no side effects.

        Confirms that triggering another event without observers leaves state unchanged.
        """
        # Arrange
        spy = []

        @EventPublisher.register(Event.test_event1)
        def any_function(any_argument):
            spy.append(f"{any_argument} has been passed")

        # Act
        EventPublisher.trigger(Event.test_event2, any_argument="any_value")

        # Assert
        assert spy == []

    def test_plugin_keep_data_between_events(self):
        """Triggering an event with a global variable.

        Confirms that a plugin can keep data between events using a global variable.
        """
        # Arrange
        plugins = {"fake_plugin": {}}
        register_plugins(plugins, source_package="tests.unit_tests")
        spy = []

        # Act
        EventPublisher.trigger(Event.test_event1, data_to_keep="any_data")
        EventPublisher.trigger(Event.test_event2, spy=spy)

        # Assert
        assert "any_data" in spy

    def test_deregistered_observers_are_not_called(self):
        """Deregister all observers and confirm that no observers are called.

        This test confirms that the `EventPublisher` can clear all registered observers,
        and that triggering events after deregistration does not call any observers.
        """
        # Arrange
        spy = []

        @EventPublisher.register(Event.test_event1)
        def any_function(any_argument):
            spy.append(f"{any_argument} has been passed")

        # Act
        EventPublisher.deregister_all()
        EventPublisher.trigger(Event.test_event1, "any_value")

        # Assert
        assert spy == []
