import asyncio
import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Set, TypeVar

from lion.core.communication.message import Note
from lion.core.forms.form import OperativeForm
from lion.protocols.operatives.reason import ReasonModel

T = TypeVar("T")


class ExecutionStrategy(Enum):
    """Execution strategy for composite operations."""

    SEQUENTIAL = "sequential"  # Execute steps in sequence
    PARALLEL = "parallel"  # Execute independent steps in parallel
    ADAPTIVE = "adaptive"  # Automatically choose between sequential and parallel


class StepStatus(Enum):
    """Status of a composite step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class CompositeStep:
    """Wrapper for operative forms with additional metadata and control."""

    def __init__(
        self,
        form: OperativeForm,
        name: str,
        retry_count: int = 3,
        timeout: float = 60.0,
    ):
        self.form = form
        self.name = name
        self.retry_count = retry_count
        self.timeout = timeout
        self.status = StepStatus.PENDING
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.error: Exception | None = None
        self.attempt = 0
        self.result: Any = None

    @property
    def execution_time(self) -> float | None:
        """Calculate execution time in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class CompositeForm(OperativeForm, Generic[T]):
    """Enhanced form for composite operations with advanced execution control and monitoring."""

    operation_type = "composite"
    context: Note
    confidence_score: float
    reasoning: ReasonModel

    def __init__(
        self,
        steps: list[CompositeStep],
        context_mapping: dict[str, Any] | None = None,
        dependencies: dict[str, list[str]] | None = None,
        guidance: str | None = None,
        execution_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
        rollback_on_failure: bool = True,
        aggregate_results: bool = True,
    ):
        """Initialize the composite form with enhanced parameters.

        Args:
            steps: List of composite steps to execute
            context_mapping: Mapping of operation types to contexts
            dependencies: Mapping of step dependencies
            guidance: Optional guidance for execution
            execution_strategy: Strategy for executing steps
            rollback_on_failure: Whether to rollback on failure
            aggregate_results: Whether to aggregate results
        """
        super().__init__()
        self.steps = steps
        self.context_mapping = context_mapping or {}
        self.dependencies = dependencies or {}
        self.guidance = guidance
        self.execution_strategy = execution_strategy
        self.rollback_on_failure = rollback_on_failure
        self.aggregate_results = aggregate_results
        self.result: list[Any] | None = None
        self.metrics: dict[str, Any] = {}

    async def execute(self) -> list[Any]:
        """Execute the composite operation with enhanced control and monitoring.

        Returns:
            List[Any]: Results from all steps

        Raises:
            ValueError: If step validation fails
            RuntimeError: If execution fails
        """
        try:
            start_time = datetime.now()
            self.metrics["start_time"] = start_time.isoformat()

            # Validate steps and dependencies
            self._validate_steps()
            dependency_graph = self._build_dependency_graph()

            # Execute based on strategy
            if self.execution_strategy == ExecutionStrategy.PARALLEL:
                results = await self._execute_parallel(dependency_graph)
            elif self.execution_strategy == ExecutionStrategy.ADAPTIVE:
                results = await self._execute_adaptive(dependency_graph)
            else:  # Sequential
                results = await self._execute_sequential()

            # Aggregate results if needed
            if self.aggregate_results:
                results = self._aggregate_results(results)

            end_time = datetime.now()
            self.metrics.update(
                {
                    "end_time": end_time.isoformat(),
                    "execution_time": (end_time - start_time).total_seconds(),
                    "successful_steps": len(
                        [s for s in self.steps if s.status == StepStatus.COMPLETED]
                    ),
                    "failed_steps": len(
                        [s for s in self.steps if s.status == StepStatus.FAILED]
                    ),
                }
            )

            self.result = results
            return self.result

        except Exception as e:
            if self.rollback_on_failure:
                await self._rollback()
            self.metrics["error"] = str(e)
            raise RuntimeError(f"Composite operation failed: {str(e)}")

    def _validate_steps(self) -> None:
        """Validate steps and their dependencies.

        Raises:
            ValueError: If validation fails
        """
        step_names = {step.name for step in self.steps}

        # Check for duplicate names
        if len(step_names) != len(self.steps):
            raise ValueError("Duplicate step names found")

        # Validate dependencies
        for step_name, deps in self.dependencies.items():
            if step_name not in step_names:
                raise ValueError(f"Unknown step in dependencies: {step_name}")
            for dep in deps:
                if dep not in step_names:
                    raise ValueError(f"Unknown dependency: {dep}")

        # Check for circular dependencies
        self._check_circular_dependencies(self.dependencies)

    def _check_circular_dependencies(self, dependencies: dict[str, list[str]]) -> None:
        """Check for circular dependencies in the step graph.

        Args:
            dependencies: Dependency mapping

        Raises:
            ValueError: If circular dependencies are found
        """
        visited = set()
        path = set()

        def visit(node: str) -> None:
            if node in path:
                raise ValueError(f"Circular dependency detected: {node}")
            if node in visited:
                return

            path.add(node)
            for dep in dependencies.get(node, []):
                visit(dep)
            path.remove(node)
            visited.add(node)

        for node in dependencies:
            visit(node)

    def _build_dependency_graph(self) -> dict[str, set[str]]:
        """Build a graph of step dependencies.

        Returns:
            Dict[str, Set[str]]: Dependency graph
        """
        graph = {}
        for step in self.steps:
            graph[step.name] = set(self.dependencies.get(step.name, []))
        return graph

    async def _execute_sequential(self) -> list[Any]:
        """Execute steps sequentially.

        Returns:
            List[Any]: Results from all steps
        """
        results = []
        for step in self.steps:
            result = await self._execute_step(step)
            results.append(result)
        return results

    async def _execute_parallel(
        self, dependency_graph: dict[str, set[str]]
    ) -> list[Any]:
        """Execute independent steps in parallel.

        Args:
            dependency_graph: Graph of step dependencies

        Returns:
            List[Any]: Results from all steps
        """
        results = []
        remaining_steps = self.steps.copy()

        while remaining_steps:
            # Find steps with no pending dependencies
            ready_steps = [
                step
                for step in remaining_steps
                if not dependency_graph[step.name]
                or all(
                    dep_name not in [s.name for s in remaining_steps]
                    for dep_name in dependency_graph[step.name]
                )
            ]

            if not ready_steps:
                raise RuntimeError("Deadlock detected in parallel execution")

            # Execute ready steps in parallel
            tasks = [self._execute_step(step) for step in ready_steps]
            step_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for step, result in zip(ready_steps, step_results):
                if isinstance(result, Exception):
                    step.status = StepStatus.FAILED
                    step.error = result
                    if self.rollback_on_failure:
                        await self._rollback()
                    raise result
                results.append(result)
                remaining_steps.remove(step)

        return results

    async def _execute_adaptive(
        self, dependency_graph: dict[str, set[str]]
    ) -> list[Any]:
        """Adaptively choose between sequential and parallel execution.

        Args:
            dependency_graph: Graph of step dependencies

        Returns:
            List[Any]: Results from all steps
        """
        # Analyze dependency graph to determine execution strategy
        total_deps = sum(len(deps) for deps in dependency_graph.values())
        if total_deps < len(self.steps) / 2:
            return await self._execute_parallel(dependency_graph)
        return await self._execute_sequential()

    async def _execute_step(self, step: CompositeStep) -> Any:
        """Execute a single step with retry logic and timeout.

        Args:
            step: Step to execute

        Returns:
            Any: Step result

        Raises:
            RuntimeError: If step execution fails
        """
        step.start_time = datetime.now()
        step.status = StepStatus.RUNNING

        # Map context if necessary
        if self.context_mapping.get(step.form.operation_type):
            step.form.context = self.context_mapping[step.form.operation_type]

        # Handle dependencies
        self._handle_step_dependencies(step)

        while step.attempt < step.retry_count:
            try:
                step.attempt += 1
                # Execute with timeout
                result = await asyncio.wait_for(
                    step.form.execute(), timeout=step.timeout
                )
                step.status = StepStatus.COMPLETED
                step.result = result
                step.end_time = datetime.now()
                return result
            except Exception as e:
                step.error = e
                if step.attempt >= step.retry_count:
                    step.status = StepStatus.FAILED
                    step.end_time = datetime.now()
                    raise RuntimeError(f"Step {step.name} failed: {str(e)}")

    def _handle_step_dependencies(self, step: CompositeStep) -> None:
        """Handle dependencies for a step.

        Args:
            step: Step to handle dependencies for
        """
        if step.name in self.dependencies:
            dependency_results = {}
            for dep_name in self.dependencies[step.name]:
                dep_step = next(s for s in self.steps if s.name == dep_name)
                if dep_step.result is not None:
                    dependency_results[dep_name] = dep_step.result
            step.form.kwargs.update(dependency_results)

    async def _rollback(self) -> None:
        """Rollback completed steps in reverse order."""
        for step in reversed(self.steps):
            if step.status == StepStatus.COMPLETED and hasattr(step.form, "rollback"):
                try:
                    await step.form.rollback()
                    step.status = StepStatus.ROLLED_BACK
                except Exception as e:
                    print(f"Rollback failed for step {step.name}: {str(e)}")

    def _aggregate_results(self, results: list[Any]) -> Any:
        """Aggregate results from all steps.

        Args:
            results: Results to aggregate

        Returns:
            Any: Aggregated results
        """
        # Implement custom aggregation logic here
        # Default implementation returns the list of results
        return results

    def save_session(self, filepath: str) -> None:
        """Save the composite session state and metrics.

        Args:
            filepath: Path to save the session data
        """
        session_data = {
            "metrics": self.metrics,
            "steps": [
                {
                    "name": step.name,
                    "status": step.status.value,
                    "execution_time": step.execution_time,
                    "attempt": step.attempt,
                    "error": str(step.error) if step.error else None,
                }
                for step in self.steps
            ],
            "parameters": {
                "execution_strategy": self.execution_strategy.value,
                "rollback_on_failure": self.rollback_on_failure,
                "aggregate_results": self.aggregate_results,
            },
        }

        with open(filepath, "w") as f:
            json.dump(session_data, f, indent=2)

    def load_session(self, filepath: str) -> None:
        """Load a previously saved composite session.

        Args:
            filepath: Path to load the session data from
        """
        with open(filepath) as f:
            session_data = json.load(f)

        self.metrics = session_data.get("metrics", {})
        params = session_data.get("parameters", {})
        self.execution_strategy = ExecutionStrategy(
            params.get("execution_strategy", ExecutionStrategy.SEQUENTIAL.value)
        )
        self.rollback_on_failure = params.get("rollback_on_failure", True)
        self.aggregate_results = params.get("aggregate_results", True)
