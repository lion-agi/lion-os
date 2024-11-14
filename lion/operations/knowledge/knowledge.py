import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Set, TypeVar

from lion.core.communication.message import Note
from lion.core.forms.form import OperativeForm
from lion.core.session.branch import Branch
from lion.core.session.session import Session
from lion.core.types import ID
from lion.protocols.operatives.instruct import INSTRUCT_MODEL_FIELD, InstructModel
from lion.protocols.operatives.reason import ReasonModel

from .prompt import PROMPT

T = TypeVar("T")


class KnowledgeSource(Enum):
    """Sources of knowledge."""

    SYSTEM = "system"
    USER = "user"
    DERIVED = "derived"
    EXTERNAL = "external"


class KnowledgeStatus(Enum):
    """Status of knowledge entries."""

    VALID = "valid"
    PENDING = "pending"
    DEPRECATED = "deprecated"
    INVALID = "invalid"


@dataclass
class KnowledgeMetadata:
    """Metadata for knowledge entries."""

    created_at: datetime
    updated_at: datetime
    source: KnowledgeSource
    version: str
    status: KnowledgeStatus
    confidence_score: float
    references: list[str]
    dependencies: set[str]


class KnowledgeBase:
    """Manager for knowledge storage and retrieval."""

    def __init__(self):
        self._storage: dict[str, Any] = {}
        self._metadata: dict[str, KnowledgeMetadata] = {}
        self._relationships: dict[str, set[str]] = {}
        self._cache: dict[str, dict[str, Any]] = {}

    def store(
        self,
        key: str,
        value: Any,
        source: KnowledgeSource,
        references: list[str] = None,
        dependencies: set[str] = None,
    ) -> None:
        """Store knowledge with metadata.

        Args:
            key: Knowledge identifier
            value: Knowledge content
            source: Source of the knowledge
            references: Related references
            dependencies: Knowledge dependencies
        """
        now = datetime.now()
        version = self._generate_version(value)

        self._storage[key] = value
        self._metadata[key] = KnowledgeMetadata(
            created_at=now,
            updated_at=now,
            source=source,
            version=version,
            status=KnowledgeStatus.VALID,
            confidence_score=1.0,
            references=references or [],
            dependencies=dependencies or set(),
        )

        # Update relationships
        if dependencies:
            for dep in dependencies:
                if dep not in self._relationships:
                    self._relationships[dep] = set()
                self._relationships[dep].add(key)

    def retrieve(self, key: str) -> Any | None:
        """Retrieve knowledge by key.

        Args:
            key: Knowledge identifier

        Returns:
            Optional[Any]: Retrieved knowledge or None
        """
        if key in self._storage:
            metadata = self._metadata[key]
            if metadata.status == KnowledgeStatus.VALID:
                return self._storage[key]
        return None

    def update(
        self,
        key: str,
        value: Any,
        source: KnowledgeSource,
        references: list[str] = None,
    ) -> None:
        """Update existing knowledge.

        Args:
            key: Knowledge identifier
            value: New knowledge content
            source: Source of the update
            references: Updated references
        """
        if key in self._storage:
            metadata = self._metadata[key]
            metadata.updated_at = datetime.now()
            metadata.version = self._generate_version(value)
            metadata.source = source
            if references:
                metadata.references = references
            self._storage[key] = value

            # Invalidate cache
            if key in self._cache:
                del self._cache[key]

    def invalidate(self, key: str) -> None:
        """Invalidate knowledge entry.

        Args:
            key: Knowledge identifier
        """
        if key in self._metadata:
            self._metadata[key].status = KnowledgeStatus.INVALID

            # Invalidate dependent knowledge
            if key in self._relationships:
                for dependent in self._relationships[key]:
                    self.invalidate(dependent)

    @staticmethod
    def _generate_version(value: Any) -> str:
        """Generate version hash for knowledge content.

        Args:
            value: Knowledge content

        Returns:
            str: Version hash
        """
        content = str(value)
        return hashlib.md5(content.encode()).hexdigest()


async def run_knowledge(
    ins: InstructModel,
    session: Session,
    branch: Branch,
    knowledge_base: KnowledgeBase | None = None,
    verbose: bool = False,
    **kwargs: Any,
) -> Any:
    """Execute a knowledge operation within the session.

    Args:
        ins: The instruction model to run.
        session: The current session.
        branch: The branch to operate on.
        knowledge_base: Optional knowledge base to use
        verbose: Whether to enable verbose output.
        **kwargs: Additional keyword arguments.

    Returns:
        The result of the knowledge operation.
    """
    if verbose:
        guidance_preview = (
            ins.guidance[:100] + "..." if len(ins.guidance) > 100 else ins.guidance
        )
        print(f"Running knowledge operation: {guidance_preview}")

    config = {**ins.model_dump(), **kwargs}
    res = await branch.operate(**config)
    branch.msgs.logger.dump()

    # Store result in knowledge base if provided
    if knowledge_base and hasattr(res, "key"):
        knowledge_base.store(
            res.key, res, KnowledgeSource.DERIVED, references=[ins.guidance]
        )

    return res


async def knowledge(
    instruct: InstructModel | dict[str, Any],
    session: Session | None = None,
    branch: Branch | ID.Ref | None = None,
    auto_run: bool = True,
    branch_kwargs: dict[str, Any] | None = None,
    return_session: bool = False,
    verbose: bool = False,
    knowledge_base: KnowledgeBase | None = None,
    **kwargs: Any,
) -> Any:
    """Perform a knowledge operation with enhanced knowledge management.

    Args:
        instruct: Instruction model or dictionary.
        session: Existing session or None to create a new one.
        branch: Existing branch or reference.
        auto_run: If True, automatically run nested instructions.
        branch_kwargs: Additional arguments for branch creation.
        return_session: If True, return the session with results.
        verbose: Whether to enable verbose output.
        knowledge_base: Optional knowledge base to use
        **kwargs: Additional keyword arguments.

    Returns:
        The results of the knowledge operation, optionally with the session.
    """
    if verbose:
        print("Starting knowledge operation.")

    field_models: list = kwargs.get("field_models", [])
    if INSTRUCT_MODEL_FIELD not in field_models:
        field_models.append(INSTRUCT_MODEL_FIELD)
    kwargs["field_models"] = field_models

    session = session or Session()
    branch = branch or session.new_branch(**(branch_kwargs or {}))

    if isinstance(instruct, InstructModel):
        instruct = instruct.clean_dump()
    if not isinstance(instruct, dict):
        raise ValueError(
            "instruct needs to be an InstructModel object or a dictionary of valid parameters"
        )

    guidance = instruct.get("guidance", "")
    instruct["guidance"] = f"\n{PROMPT}\n{guidance}"

    res1 = await branch.operate(**instruct, **kwargs)
    if verbose:
        print("Initial knowledge operation complete.")

    if not auto_run:
        if return_session:
            return res1, session
        return res1

    results = res1 if isinstance(res1, list) else [res1]
    if hasattr(res1, "instruct_models"):
        instructs: list[InstructModel] = res1.instruct_models
        for i, ins in enumerate(instructs, 1):
            if verbose:
                print(f"\nExecuting knowledge step {i}/{len(instructs)}")
            res = await run_knowledge(
                ins,
                session,
                branch,
                knowledge_base=knowledge_base,
                verbose=verbose,
                **kwargs,
            )
            results.append(res)

        if verbose:
            print("\nAll knowledge steps completed successfully!")
    if return_session:
        return results, session
    return results


class KnowledgeForm(OperativeForm, Generic[T]):
    """Enhanced form for knowledge operations with advanced knowledge management."""

    operation_type = "knowledge"
    context: Note
    confidence_score: float
    reasoning: ReasonModel

    def __init__(
        self,
        context: Note,
        guidance: str | None = None,
        knowledge_base: KnowledgeBase | None = None,
        min_confidence: float = 0.7,
        cache_enabled: bool = True,
        cache_ttl: int = 3600,  # Cache time-to-live in seconds
        validate_knowledge: bool = True,
    ):
        """Initialize the knowledge form with enhanced parameters.

        Args:
            context: The context for knowledge operations
            guidance: Optional guidance for operations
            knowledge_base: Optional knowledge base to use
            min_confidence: Minimum confidence score
            cache_enabled: Whether to enable caching
            cache_ttl: Cache time-to-live in seconds
            validate_knowledge: Whether to validate knowledge
        """
        super().__init__()
        self.context = context
        self.guidance = guidance
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.min_confidence = min_confidence
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.validate_knowledge = validate_knowledge
        self.result: Any | None = None
        self.metrics: dict[str, Any] = {}

    async def execute(self) -> Any:
        """Execute the knowledge operation with enhanced management.

        Returns:
            Any: Knowledge operation result

        Raises:
            ValueError: If knowledge validation fails
            RuntimeError: If operation execution fails
        """
        try:
            start_time = datetime.now()

            # Check cache if enabled
            cache_key = self._generate_cache_key()
            if self.cache_enabled and self._check_cache(cache_key):
                self.metrics["cache_hit"] = True
                return self._get_cached_result(cache_key)

            # Create instruction
            instruct = InstructModel(
                instruction=self.guidance
                or "Please perform a knowledge operation based on the context",
                context=self.context,
            )

            # Execute knowledge operation
            knowledge_result = await knowledge(
                instruct=instruct,
                auto_run=False,
                verbose=False,
                knowledge_base=self.knowledge_base,
            )

            # Validate result if enabled
            if self.validate_knowledge:
                self._validate_knowledge(knowledge_result)

            # Update metrics
            self.metrics.update(
                {
                    "execution_time": (datetime.now() - start_time).total_seconds(),
                    "cache_hit": False,
                    "knowledge_base_size": len(self.knowledge_base._storage),
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Cache result if enabled
            if self.cache_enabled:
                self._cache_result(cache_key, knowledge_result)

            self.result = knowledge_result
            return self.result

        except Exception as e:
            self.metrics["error"] = str(e)
            raise RuntimeError(f"Knowledge operation failed: {str(e)}")

    def _generate_cache_key(self) -> str:
        """Generate cache key based on context and guidance.

        Returns:
            str: Cache key
        """
        content = f"{str(self.context)}{self.guidance or ''}"
        return hashlib.md5(content.encode()).hexdigest()

    def _check_cache(self, key: str) -> bool:
        """Check if valid cache entry exists.

        Args:
            key: Cache key

        Returns:
            bool: Whether valid cache exists
        """
        if not hasattr(self.knowledge_base, "_cache"):
            return False

        if key not in self.knowledge_base._cache:
            return False

        cache_entry = self.knowledge_base._cache[key]
        cache_time = datetime.fromisoformat(cache_entry["timestamp"])

        if (datetime.now() - cache_time).total_seconds() > self.cache_ttl:
            del self.knowledge_base._cache[key]
            return False

        return True

    def _get_cached_result(self, key: str) -> Any:
        """Get cached result.

        Args:
            key: Cache key

        Returns:
            Any: Cached result
        """
        return self.knowledge_base._cache[key]["result"]

    def _cache_result(self, key: str, result: Any) -> None:
        """Cache operation result.

        Args:
            key: Cache key
            result: Result to cache
        """
        self.knowledge_base._cache[key] = {
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }

    def _validate_knowledge(self, result: Any) -> None:
        """Validate knowledge operation result.

        Args:
            result: Result to validate

        Raises:
            ValueError: If validation fails
        """
        if not result:
            raise ValueError("Empty knowledge operation result")

        if hasattr(result, "confidence_score"):
            if result.confidence_score < self.min_confidence:
                raise ValueError(
                    f"Knowledge confidence score {result.confidence_score} "
                    f"below threshold {self.min_confidence}"
                )

    def save_session(self, filepath: str) -> None:
        """Save the knowledge session state and metrics.

        Args:
            filepath: Path to save the session data
        """
        session_data = {
            "metrics": self.metrics,
            "knowledge_base": {
                "size": len(self.knowledge_base._storage),
                "valid_entries": len(
                    [
                        k
                        for k, m in self.knowledge_base._metadata.items()
                        if m.status == KnowledgeStatus.VALID
                    ]
                ),
                "relationships": len(self.knowledge_base._relationships),
            },
            "parameters": {
                "min_confidence": self.min_confidence,
                "cache_enabled": self.cache_enabled,
                "cache_ttl": self.cache_ttl,
                "validate_knowledge": self.validate_knowledge,
            },
        }

        with open(filepath, "w") as f:
            json.dump(session_data, f, indent=2)

    def load_session(self, filepath: str) -> None:
        """Load a previously saved knowledge session.

        Args:
            filepath: Path to load the session data from
        """
        with open(filepath) as f:
            session_data = json.load(f)

        self.metrics = session_data.get("metrics", {})
        params = session_data.get("parameters", {})
        self.min_confidence = params.get("min_confidence", 0.7)
        self.cache_enabled = params.get("cache_enabled", True)
        self.cache_ttl = params.get("cache_ttl", 3600)
        self.validate_knowledge = params.get("validate_knowledge", True)
