from collections.abc import Callable
from typing import Any, Dict, List, Union

from lion.core.models import FieldModel, NewModelParams
from lion.core.session.branch import Branch
from lion.core.session.session import Session
from lion.core.types import ID
from lion.libs.func import alcall
from lion.protocols.operatives.instruct import InstructModel

from .base import BaseConfig, OperationType, OperativeForm, PatternMatcher, ReasonModel


class CompositeConfig(BaseConfig):
    operations: list[dict[str, Any]]
    auto_execute: bool = True
    track_patterns: bool = True


class CompositeOperation(OperativeForm[dict[str, Any]]):
    """Enhanced composite operation with Operative integration."""

    def __init__(
        self,
        operations: list[dict[str, Any]],
        session: Session | None = None,
        branch: Branch | ID.Ref | None = None,
        branch_kwargs: dict[str, Any] | None = None,
        verbose: bool = False,
        request_params: NewModelParams | None = None,
        field_models: list[FieldModel] | None = None,
        exclude_fields: list[str] | None = None,
    ):
        config = CompositeConfig(
            operation_type=OperationType.COMPOSITE, operations=operations, context={}
        )
        super().__init__(config)
        self.operations = operations
        self.session = session or Session()
        self.branch = branch
        self.branch_kwargs = branch_kwargs or {}
        self.verbose = verbose
        self.results = {}
        self.pattern_matcher = PatternMatcher()

        if request_params or field_models or exclude_fields:
            self.setup_operative(
                request_params=request_params,
                field_models=field_models,
                exclude_fields=exclude_fields,
            )

    async def execute_operation(
        self, operation: dict[str, Any], prev_results: dict[str, Any]
    ) -> Any:
        """Execute with pattern matching and tracking."""
        op_type = operation["type"]
        op_config = operation.get("config", {})

        # Insert previous results into config if dependencies specified
        deps = operation.get("depends_on", [])
        for dep in deps:
            if dep in prev_results:
                op_config["context"] = prev_results[dep]

        # Check for similar patterns
        pattern_match = self.pattern_matcher.find_similar_pattern(operation)
        if pattern_match:
            pattern_id, confidence = pattern_match
            self.reasoning.pattern_matches.append(pattern_id)
            self.reasoning.confidence = max(self.reasoning.confidence, confidence)

        if op_type == "select":
            from ..select import select

            result = await select(
                instruct=op_config,
                branch=self.branch,
                branch_kwargs=self.branch_kwargs,
                verbose=self.verbose,
            )
        elif op_type == "plan":
            from ..plan import plan

            result = await plan(
                instruct=op_config,
                session=self.session,
                branch=self.branch,
                branch_kwargs=self.branch_kwargs,
                verbose=self.verbose,
            )
        elif op_type == "brainstorm":
            from ..brainstorm import brainstorm

            result = await brainstorm(
                instruct=op_config,
                session=self.session,
                branch=self.branch,
                branch_kwargs=self.branch_kwargs,
                verbose=self.verbose,
            )
        else:
            raise ValueError(f"Unknown operation type: {op_type}")

        # Track successful patterns
        if result and self.config.track_patterns:
            self.pattern_matcher.add_pattern(
                f"{op_type}_{len(self.pattern_matcher.patterns)}",
                operation,
                1.0,  # Success score can be refined
            )

        # Track result in operative if available
        if result and self.operative:
            self.operative.update_response_model(
                data={op_type: result} if isinstance(result, dict) else None,
                text=result if isinstance(result, str) else None,
            )

        return result

    async def execute(self) -> dict[str, Any]:
        """Execute all operations respecting dependencies."""
        for op in self.operations:
            op_name = op.get("name", f"op_{len(self.results)}")
            if self.verbose:
                print(f"Executing operation: {op_name}")

            result = await self.execute_operation(op, self.results)
            self.results[op_name] = result

        return self.results

    def validate(self) -> bool:
        """Validate composite operation configuration."""
        if not self.config.operations:
            return False
        # Validate dependencies
        deps = set()
        for op in self.config.operations:
            if "depends_on" in op:
                deps.update(op["depends_on"])
        return deps.issubset({op.get("name") for op in self.config.operations})


async def compose(
    operations: list[dict[str, Any]],
    session: Session | None = None,
    branch: Branch | ID.Ref | None = None,
    branch_kwargs: dict[str, Any] | None = None,
    return_session: bool = False,
    verbose: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], Session]:
    """Enhanced compose with form validation and pattern learning."""
    composite = CompositeOperation(
        operations=operations,
        session=session,
        branch=branch,
        branch_kwargs=branch_kwargs,
        verbose=verbose,
    )

    if not composite.validate():
        raise ValueError("Invalid composite operation configuration")

    results = await composite.execute()

    if return_session:
        return results, composite.session
    return results
