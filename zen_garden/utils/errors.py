class OptimizationError(RuntimeError):
    """Exception raised when the optimization problem is infeasible."""

    def __init__(
        self,
        status="The optimization is infeasible or unbounded, or finished with an error",
    ):
        """Initializes the class.

        :param status: The message to display
        """
        self.message = f"The termination condition was {status}"
        super().__init__(self.message)
