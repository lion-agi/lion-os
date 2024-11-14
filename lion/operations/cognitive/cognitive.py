import hashlib
import json
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar

from lion.core.communication.message import Note
from lion.core.forms.form import OperativeForm
from lion.core.session.branch import Branch
from lion.core.session.session import Session
from lion.core.types import ID
from lion.libs.func import alcall
from lion.protocols.operatives.instruct import INSTRUCT_MODEL_FIELD, InstructModel
from lion.protocols.operatives.reason import ReasonModel

from .prompt import PROMPT

T = TypeVar("T")


async def run_cognitive_process(
    ins: InstructModel,
    session: Session,
    branch: Branch,
    auto_run: bool,
    verbose: bool = False,
    **kwargs: Any,
) -> Any:
    """Execute a cognitive process within the session.

    Args:
        ins: The instruction model to run.
        session: The current session.
        branch: The branch to operate on.
        auto_run: Whether to automatically run nested instructions.
        verbose: Whether to enable verbose output.
        **kwargs: Additional keyword arguments.

    Returns:
        The result of the cognitive process execution.
    """
    if verbose:
        guidance_preview = (
            ins.guidance[:100] + "..." if len(ins.guidance) > 100 else ins.guidance
        )
        print(f"Running cognitive process: {guidance_preview}")

    async def run(ins_):
        b_ = session.split(branch)
        return await run_cognitive_process(ins_, session, b_, False, **kwargs)

    config = {**ins.model_dump(), **kwargs}
    res = await branch.operate(**config)
    branch.msgs.logger.dump()
    instructs = []

    if hasattr(res, "instruct_models"):
        instructs = res.instruct_models

    if auto_run is True and instructs:
        ress = await alcall(instructs, run)
        response_ = []
        for res in ress:
            if isinstance(res, list):
                response_.extend(res)
            else:
                response_.append(res)
        response_.insert(0, res)
        return response_

    return res


async def cognitive(
    instruct: InstructModel | dict[str, Any],
    session: Session | None = None,
    branch: Branch | ID.Ref | None = None,
    auto_run: bool = True,
    branch_kwargs: dict[str, Any] | None = None,
    return_session: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> Any:
    """Perform a cognitive operation.

    Args:
        instruct: Instruction model or dictionary.
        session: Existing session or None to create a new one.
        branch: Existing branch or reference.
        auto_run: If True, automatically run generated instructions.
        branch_kwargs: Additional arguments for branch creation.
        return_session: If True, return the session with results.
        verbose: Whether to enable verbose output.
        **kwargs: Additional keyword arguments.

    Returns:
        The results of the cognitive operation, optionally with the session.
    """
    if verbose:
        print("Starting cognitive operation.")

    field_models: list = kwargs.get("field_models", [])
    if INSTRUCT_MODEL_FIELD not in field_models:
        field_models.append(INSTRUCT_MODEL_FIELD)

    kwargs["field_models"] = field_models

    if session is not None:
        if branch is not None:
            branch: Branch = session.branches[branch]
        else:
            branch = session.new_branch(**(branch_kwargs or {}))
    else:
        session = Session()
        if isinstance(branch, Branch):
            session.branches.include(branch)
            session.default_branch = branch
        if branch is None:
            branch = session.new_branch(**(branch_kwargs or {}))

    if isinstance(instruct, InstructModel):
        instruct = instruct.clean_dump()
    if not isinstance(instruct, dict):
        raise ValueError(
            "instruct needs to be an InstructModel obj or a dictionary of valid parameters"
        )

    guidance = instruct.get("guidance", "")
    instruct["guidance"] = f"\n{PROMPT}" + guidance

    res1 = await branch.operate(**instruct, **kwargs)
    if verbose:
        print("Initial cognitive operation complete.")

    instructs = None

    async def run(ins_):
        b_ = session.split(branch)
        return await run_cognitive_process(
            ins_, session, b_, auto_run, verbose=verbose, **kwargs
        )

    if not auto_run:
        return res1

    async with session.branches:
        if hasattr(res1, "instruct_models"):
            instructs: list[InstructModel] = res1.instruct_models
            ress = await alcall(instructs, run)
            response_ = []

            for res in ress:
                if isinstance(res, list):
                    response_.extend(res)
                else:
                    response_.append(res)

            response_.insert(0, res1)
            if return_session:
                return response_, session
            return response_

    if return_session:
        return res1, session

    return res1


class CognitiveForm(OperativeForm, Generic[T]):
    """Enhanced form for cognitive operations with caching, validation, and metrics tracking."""

    operation_type = "cognitive"
    context: Note
    confidence_score: float
    reasoning: ReasonModel

    def __init__(
        self,
        context: Note,
        guidance: str | None = None,
        min_confidence_score: float = 0.7,
        cache_results: bool = True,
        cache_ttl: int = 3600,  # Cache time-to-live in seconds
        validation_threshold: float = 0.8,
    ):
        """Initialize the cognitive form with enhanced parameters.

        Args:
            context: The context for cognitive analysis
            guidance: Optional guidance for the analysis
            min_confidence_score: Minimum confidence score for valid results
            cache_results: Whether to cache results
            cache_ttl: Time-to-live for cached results in seconds
            validation_threshold: Threshold for result validation
        """
        super().__init__()
        self.context = context
        self.guidance = guidance
        self.result: Any | None = None
        self.min_confidence_score = min_confidence_score
        self.cache_results = cache_results
        self.cache_ttl = cache_ttl
        self.validation_threshold = validation_threshold
        self.metrics: dict[str, Any] = {}
        self._cache: dict[str, dict[str, Any]] = {}

    async def execute(self) -> Any:
        """Execute the cognitive process with enhanced error handling and metrics.

        Returns:
            Any: The validated cognitive analysis result

        Raises:
            ValueError: If the context or parameters are invalid
            RuntimeError: If the cognitive process fails
        """
        try:
            # Start metrics tracking
            start_time = datetime.now()

            # Validate inputs
            if not self.context or not isinstance(self.context, Note):
                raise ValueError("Invalid context provided")

            # Check cache first
            cache_key = self._generate_cache_key()
            if self.cache_results and self._check_cache(cache_key):
                self.metrics["cache_hit"] = True
                return self._cache[cache_key]["result"]

            # Generate instruction
            instruct = InstructModel(
                instruction=self.guidance
                or "Please perform a cognitive analysis based on the context",
                context=self.context,
            )

            # Execute cognitive process
            result = await cognitive(instruct=instruct, auto_run=False, verbose=False)

            # Validate and assess result
            validated_result = self._validate_result(result)
            quality_score = self._assess_quality(validated_result)

            # Update metrics
            self.metrics.update(
                {
                    "execution_time": (datetime.now() - start_time).total_seconds(),
                    "quality_score": quality_score,
                    "cache_hit": False,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Cache result if enabled
            if self.cache_results:
                self._cache_result(cache_key, validated_result)

            self.result = validated_result
            return self.result

        except Exception as e:
            self.metrics["error"] = str(e)
            raise RuntimeError(f"Cognitive process failed: {str(e)}")

    def _generate_cache_key(self) -> str:
        """Generate a unique cache key based on context and guidance.

        Returns:
            str: Unique cache key
        """
        content = f"{str(self.context)}{self.guidance or ''}"
        return hashlib.md5(content.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> bool:
        """Check if a valid cache entry exists.

        Args:
            cache_key: The cache key to check

        Returns:
            bool: True if valid cache entry exists
        """
        if cache_key not in self._cache:
            return False

        cache_entry = self._cache[cache_key]
        cache_time = datetime.fromisoformat(cache_entry["timestamp"])

        if (datetime.now() - cache_time).total_seconds() > self.cache_ttl:
            del self._cache[cache_key]
            return False

        return True

    def _cache_result(self, cache_key: str, result: Any) -> None:
        """Cache a result with timestamp.

        Args:
            cache_key: The cache key
            result: The result to cache
        """
        self._cache[cache_key] = {
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }

    def _validate_result(self, result: Any) -> Any:
        """Validate the cognitive analysis result.

        Args:
            result: Result to validate

        Returns:
            Any: Validated result

        Raises:
            ValueError: If result fails validation
        """
        if hasattr(result, "confidence_score"):
            if result.confidence_score < self.min_confidence_score:
                raise ValueError(
                    f"Result confidence score {result.confidence_score} below threshold {self.min_confidence_score}"
                )

        if not result:
            raise ValueError("Empty result from cognitive analysis")

        return result

    def _assess_quality(self, result: Any) -> float:
        """Assess the quality of the cognitive analysis result.

        Args:
            result: Result to assess

        Returns:
            float: Quality score between 0 and 1
        """
        quality_score = 1.0

        # Assess based on confidence score if available
        if hasattr(result, "confidence_score"):
            quality_score *= result.confidence_score

        # Assess based on reasoning if available
        if hasattr(result, "reasoning") and result.reasoning:
            quality_score *= 1.0
        else:
            quality_score *= 0.8

        # Assess based on completeness
        if isinstance(result, dict):
            expected_keys = {"analysis", "conclusion"}
            actual_keys = set(result.keys())
            completeness = len(actual_keys.intersection(expected_keys)) / len(
                expected_keys
            )
            quality_score *= completeness

        return quality_score

    def save_session(self, filepath: str) -> None:
        """Save the cognitive session results and metrics.

        Args:
            filepath: Path to save the session data
        """
        session_data = {
            "metrics": self.metrics,
            "result": (
                self.result
                if not hasattr(self.result, "model_dump")
                else self.result.model_dump()
            ),
            "parameters": {
                "min_confidence_score": self.min_confidence_score,
                "cache_ttl": self.cache_ttl,
                "validation_threshold": self.validation_threshold,
            },
        }

        with open(filepath, "w") as f:
            json.dump(session_data, f, indent=2)

    def load_session(self, filepath: str) -> None:
        """Load a previously saved cognitive session.

        Args:
            filepath: Path to load the session data from
        """
        with open(filepath) as f:
            session_data = json.load(f)

        self.metrics = session_data.get("metrics", {})
        self.result = session_data.get("result")
        params = session_data.get("parameters", {})
        self.min_confidence_score = params.get(
            "min_confidence_score", self.min_confidence_score
        )
        self.cache_ttl = params.get("cache_ttl", self.cache_ttl)
        self.validation_threshold = params.get(
            "validation_threshold", self.validation_threshold
        )
