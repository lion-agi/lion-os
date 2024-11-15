from typing import Dict, Optional

from pydantic import Field

from .base import BaseConfig


class LionIDConfig(BaseConfig):
    """Configuration for ID generation and management."""

    # Basic ID settings
    prefix: str = Field(default="LID", description="ID prefix for generated IDs")
    postfix: str = Field(default="", description="ID suffix for generated IDs")
    length: int = Field(default=16, description="Length of generated IDs")

    # Hyphen settings
    separator: str = Field(default="-", description="Separator for ID components")
    random_hyphen: bool = Field(
        default=False, description="Use random hyphen placement"
    )
    num_hyphens: int = Field(default=3, description="Number of hyphens to include")
    hyphen_start_index: int = Field(
        default=4, description="Start index for hyphen placement"
    )
    hyphen_end_index: int = Field(
        default=-4, description="End index for hyphen placement"
    )

    # Legacy compatibility
    n: int = Field(default=36, description="Legacy length parameter")

    # Optional custom ID formats for different entity types
    entity_formats: dict[str, str] = Field(
        default_factory=dict, description="Custom ID formats for different entity types"
    )

    # Security settings
    require_unique: bool = Field(default=True, description="Enforce ID uniqueness")
    validate_format: bool = Field(default=True, description="Validate ID format")

    def __init__(self, **data):
        super().__init__(**data)
        if self.is_production:
            # Force security settings in production
            self.require_unique = True
            self.validate_format = True

    def get_entity_format(self, entity_type: str) -> str | None:
        """Get custom ID format for an entity type."""
        return self.entity_formats.get(entity_type)

    def validate_security(self) -> None:
        """Validate security-critical configuration settings."""
        super().validate_security()
        if self.is_production:
            if not self.require_unique:
                raise ValueError("ID uniqueness must be enforced in production")
            if not self.validate_format:
                raise ValueError("ID format validation must be enabled in production")
            if self.length < 12:
                raise ValueError(
                    "ID length must be at least 12 characters in production"
                )
            if self.random_hyphen:
                raise ValueError("Random hyphen placement is not allowed in production")
