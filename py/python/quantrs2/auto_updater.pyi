"""Type stubs for auto_updater module."""

from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from pathlib import Path

class UpdatePolicy(Enum):
    MANUAL: str
    NOTIFY: str
    AUTOMATIC: str
    DISABLED: str

class UpdateChannel(Enum):
    STABLE: str
    BETA: str
    ALPHA: str
    DEV: str

class VersionInfo:
    version: str
    release_date: str
    python_version: str
    size: int
    upload_url: str
    sha256: Optional[str]
    requires_python: Optional[str]

    def __init__(
        self,
        version: str,
        release_date: str,
        python_version: str,
        size: int,
        upload_url: str,
        sha256: Optional[str] = ...,
        requires_python: Optional[str] = ...
    ) -> None: ...

class UpdateInfo:
    current_version: str
    latest_version: str
    update_available: bool
    changelog_url: Optional[str]
    release_notes: Optional[str]
    is_security_update: bool
    is_critical: bool

    def __init__(
        self,
        current_version: str,
        latest_version: str,
        update_available: bool,
        changelog_url: Optional[str] = ...,
        release_notes: Optional[str] = ...,
        is_security_update: bool = ...,
        is_critical: bool = ...
    ) -> None: ...

class UpdaterConfig:
    policy: UpdatePolicy
    channel: UpdateChannel
    check_interval_days: int
    auto_check_on_import: bool
    pypi_url: str
    backup_before_update: bool

    def __init__(
        self,
        policy: UpdatePolicy = ...,
        channel: UpdateChannel = ...,
        check_interval_days: int = ...,
        auto_check_on_import: bool = ...,
        pypi_url: str = ...,
        backup_before_update: bool = ...
    ) -> None: ...

    def to_dict(self) -> Dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UpdaterConfig: ...

class QuantRS2Updater:
    PACKAGE_NAME: str
    CONFIG_FILE: Path
    config: UpdaterConfig

    def __init__(self, config: Optional[UpdaterConfig] = ...) -> None: ...
    def save_config(self) -> None: ...
    def check_for_updates(self) -> UpdateInfo: ...
    def install_update(self, version: Optional[str] = ..., force: bool = ...) -> bool: ...
    def get_version_info(self, version: str) -> Optional[VersionInfo]: ...
    def list_available_versions(self, limit: int = ...) -> List[str]: ...

def get_updater() -> QuantRS2Updater: ...
def check_for_updates() -> Tuple[bool, str]: ...
def install_update(version: Optional[str] = ..., force: bool = ...) -> bool: ...
def configure_updater(
    policy: UpdatePolicy = ...,
    channel: UpdateChannel = ...,
    **kwargs: Any
) -> None: ...
def get_current_version() -> str: ...
def list_versions(limit: int = ...) -> List[str]: ...
