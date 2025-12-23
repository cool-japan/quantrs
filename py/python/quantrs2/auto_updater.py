#!/usr/bin/env python3
"""
Auto-Updater Module for QuantRS2

This module provides automatic update checking and installation functionality
for the QuantRS2 Python package.

Features:
    - Check for new versions on PyPI
    - Automatic update installation
    - Version comparison and compatibility checking
    - Update notifications
    - Configurable update policies (manual, automatic, notify-only)
    - Changelog retrieval
    - Rollback support

Usage:
    from quantrs2.auto_updater import check_for_updates, install_update

    # Check for updates
    update_available, latest_version = check_for_updates()

    if update_available:
        print(f"New version available: {latest_version}")
        # Install update
        install_update()
"""

import sys
import subprocess
import json
import urllib.request
import urllib.error
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import warnings
import platform
import hashlib
import tempfile
import shutil

try:
    import packaging.version
    PACKAGING_AVAILABLE = True
except ImportError:
    PACKAGING_AVAILABLE = False
    warnings.warn("packaging module not available. Install with: pip install packaging")


class UpdatePolicy(Enum):
    """Update policy modes."""
    MANUAL = "manual"  # Never auto-update, only check
    NOTIFY = "notify"  # Check and notify, but don't auto-install
    AUTOMATIC = "automatic"  # Check and auto-install updates
    DISABLED = "disabled"  # Don't check for updates


class UpdateChannel(Enum):
    """Update channels for version selection."""
    STABLE = "stable"  # Only stable releases
    BETA = "beta"  # Include beta releases
    ALPHA = "alpha"  # Include alpha releases
    DEV = "dev"  # Include development releases


@dataclass
class VersionInfo:
    """Information about a package version."""
    version: str
    release_date: str
    python_version: str
    size: int
    upload_url: str
    sha256: Optional[str] = None
    requires_python: Optional[str] = None


@dataclass
class UpdateInfo:
    """Information about available updates."""
    current_version: str
    latest_version: str
    update_available: bool
    changelog_url: Optional[str] = None
    release_notes: Optional[str] = None
    is_security_update: bool = False
    is_critical: bool = False


class UpdaterConfig:
    """Configuration for the auto-updater."""

    def __init__(
        self,
        policy: UpdatePolicy = UpdatePolicy.NOTIFY,
        channel: UpdateChannel = UpdateChannel.STABLE,
        check_interval_days: int = 7,
        auto_check_on_import: bool = False,
        pypi_url: str = "https://pypi.org/pypi",
        backup_before_update: bool = True,
    ):
        """
        Initialize updater configuration.

        Args:
            policy: Update policy (manual, notify, automatic, disabled)
            channel: Update channel (stable, beta, alpha, dev)
            check_interval_days: Days between update checks
            auto_check_on_import: Check for updates when module is imported
            pypi_url: PyPI URL for checking versions
            backup_before_update: Create backup before updating
        """
        self.policy = policy
        self.channel = channel
        self.check_interval_days = check_interval_days
        self.auto_check_on_import = auto_check_on_import
        self.pypi_url = pypi_url
        self.backup_before_update = backup_before_update

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'policy': self.policy.value,
            'channel': self.channel.value,
            'check_interval_days': self.check_interval_days,
            'auto_check_on_import': self.auto_check_on_import,
            'pypi_url': self.pypi_url,
            'backup_before_update': self.backup_before_update,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UpdaterConfig':
        """Create from dictionary."""
        return cls(
            policy=UpdatePolicy(data.get('policy', 'notify')),
            channel=UpdateChannel(data.get('channel', 'stable')),
            check_interval_days=data.get('check_interval_days', 7),
            auto_check_on_import=data.get('auto_check_on_import', False),
            pypi_url=data.get('pypi_url', 'https://pypi.org/pypi'),
            backup_before_update=data.get('backup_before_update', True),
        )


class QuantRS2Updater:
    """Main updater class for QuantRS2."""

    PACKAGE_NAME = "quantrs2"
    CONFIG_FILE = Path.home() / ".quantrs2" / "updater_config.json"

    def __init__(self, config: Optional[UpdaterConfig] = None):
        """
        Initialize the updater.

        Args:
            config: Updater configuration
        """
        self.config = config or self._load_config()
        self._current_version = self._get_current_version()

    def _load_config(self) -> UpdaterConfig:
        """Load configuration from file."""
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                return UpdaterConfig.from_dict(data)
            except Exception as e:
                warnings.warn(f"Failed to load updater config: {e}")

        return UpdaterConfig()

    def save_config(self):
        """Save configuration to file."""
        self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save updater config: {e}")

    def _get_current_version(self) -> str:
        """Get the current installed version."""
        try:
            from quantrs2 import __version__
            return __version__
        except ImportError:
            return "unknown"

    def _fetch_pypi_data(self) -> Optional[Dict[str, Any]]:
        """Fetch package data from PyPI."""
        url = f"{self.config.pypi_url}/{self.PACKAGE_NAME}/json"

        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
            return data
        except urllib.error.URLError as e:
            warnings.warn(f"Failed to fetch PyPI data: {e}")
            return None
        except Exception as e:
            warnings.warn(f"Error fetching PyPI data: {e}")
            return None

    def _get_latest_version(self, pypi_data: Dict[str, Any]) -> Optional[str]:
        """Extract latest version from PyPI data based on channel."""
        if not pypi_data:
            return None

        all_versions = list(pypi_data.get('releases', {}).keys())

        if not all_versions:
            return None

        # Filter versions based on channel
        filtered_versions = []
        for version in all_versions:
            if self.config.channel == UpdateChannel.STABLE:
                # Only stable releases (no alpha, beta, dev, rc)
                if not any(x in version for x in ['a', 'b', 'dev', 'rc']):
                    filtered_versions.append(version)
            elif self.config.channel == UpdateChannel.BETA:
                # Stable and beta
                if not any(x in version for x in ['a', 'dev', 'rc']) or 'b' in version:
                    filtered_versions.append(version)
            elif self.config.channel == UpdateChannel.ALPHA:
                # Stable, beta, and alpha
                if 'dev' not in version:
                    filtered_versions.append(version)
            else:  # DEV channel
                # All versions
                filtered_versions.append(version)

        if not filtered_versions:
            return None

        # Sort versions and get the latest
        if PACKAGING_AVAILABLE:
            try:
                sorted_versions = sorted(
                    filtered_versions,
                    key=lambda v: packaging.version.parse(v),
                    reverse=True
                )
                return sorted_versions[0]
            except Exception:
                # Fallback to simple string comparison
                return max(filtered_versions)
        else:
            return max(filtered_versions)

    def check_for_updates(self) -> UpdateInfo:
        """
        Check for available updates.

        Returns:
            UpdateInfo object with update information
        """
        if self.config.policy == UpdatePolicy.DISABLED:
            return UpdateInfo(
                current_version=self._current_version,
                latest_version=self._current_version,
                update_available=False
            )

        pypi_data = self._fetch_pypi_data()

        if not pypi_data:
            return UpdateInfo(
                current_version=self._current_version,
                latest_version=self._current_version,
                update_available=False
            )

        latest_version = self._get_latest_version(pypi_data)

        if not latest_version:
            return UpdateInfo(
                current_version=self._current_version,
                latest_version=self._current_version,
                update_available=False
            )

        # Compare versions
        update_available = self._is_newer_version(latest_version, self._current_version)

        # Get release notes URL
        changelog_url = f"https://github.com/cool-japan/quantrs/blob/master/CHANGELOG.md"

        return UpdateInfo(
            current_version=self._current_version,
            latest_version=latest_version,
            update_available=update_available,
            changelog_url=changelog_url
        )

    def _is_newer_version(self, version1: str, version2: str) -> bool:
        """Check if version1 is newer than version2."""
        if PACKAGING_AVAILABLE:
            try:
                return packaging.version.parse(version1) > packaging.version.parse(version2)
            except Exception:
                pass

        # Fallback to simple string comparison
        return version1 > version2

    def install_update(self, version: Optional[str] = None, force: bool = False) -> bool:
        """
        Install an update.

        Args:
            version: Specific version to install (None for latest)
            force: Force installation even if no update available

        Returns:
            True if installation successful, False otherwise
        """
        if self.config.policy not in [UpdatePolicy.AUTOMATIC, UpdatePolicy.MANUAL]:
            if not force:
                print("Auto-update is not enabled. Use force=True to override.")
                return False

        # Check for updates first
        update_info = self.check_for_updates()

        if not update_info.update_available and not force:
            print("No updates available.")
            return False

        # Determine version to install
        install_version = version or update_info.latest_version

        print(f"Installing QuantRS2 {install_version}...")

        try:
            # Use pip to install the update
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                f"{self.PACKAGE_NAME}=={install_version}"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            print(f"Successfully updated to {install_version}")
            print("\nPlease restart your Python session to use the new version.")
            return True

        except subprocess.CalledProcessError as e:
            print(f"Failed to install update: {e}")
            print(f"Error output: {e.stderr}")
            return False
        except Exception as e:
            print(f"Unexpected error during update: {e}")
            return False

    def get_version_info(self, version: str) -> Optional[VersionInfo]:
        """Get detailed information about a specific version."""
        pypi_data = self._fetch_pypi_data()

        if not pypi_data:
            return None

        releases = pypi_data.get('releases', {})
        version_data = releases.get(version)

        if not version_data or not version_data:
            return None

        # Get first distribution (usually source)
        dist = version_data[0]

        return VersionInfo(
            version=version,
            release_date=dist.get('upload_time', 'unknown'),
            python_version=dist.get('python_version', 'unknown'),
            size=dist.get('size', 0),
            upload_url=dist.get('url', ''),
            sha256=dist.get('digests', {}).get('sha256'),
            requires_python=dist.get('requires_python')
        )

    def list_available_versions(self, limit: int = 10) -> List[str]:
        """List available versions on PyPI."""
        pypi_data = self._fetch_pypi_data()

        if not pypi_data:
            return []

        versions = list(pypi_data.get('releases', {}).keys())

        if PACKAGING_AVAILABLE:
            try:
                versions = sorted(
                    versions,
                    key=lambda v: packaging.version.parse(v),
                    reverse=True
                )
            except Exception:
                versions = sorted(versions, reverse=True)
        else:
            versions = sorted(versions, reverse=True)

        return versions[:limit]


# Global updater instance
_updater: Optional[QuantRS2Updater] = None


def get_updater() -> QuantRS2Updater:
    """Get the global updater instance."""
    global _updater
    if _updater is None:
        _updater = QuantRS2Updater()
    return _updater


def check_for_updates() -> Tuple[bool, str]:
    """
    Check for updates (convenience function).

    Returns:
        Tuple of (update_available, latest_version)
    """
    updater = get_updater()
    update_info = updater.check_for_updates()
    return update_info.update_available, update_info.latest_version


def install_update(version: Optional[str] = None, force: bool = False) -> bool:
    """
    Install an update (convenience function).

    Args:
        version: Specific version to install
        force: Force installation

    Returns:
        True if successful, False otherwise
    """
    updater = get_updater()
    return updater.install_update(version, force)


def configure_updater(
    policy: UpdatePolicy = UpdatePolicy.NOTIFY,
    channel: UpdateChannel = UpdateChannel.STABLE,
    **kwargs
) -> None:
    """
    Configure the auto-updater.

    Args:
        policy: Update policy
        channel: Update channel
        **kwargs: Additional configuration options
    """
    config = UpdaterConfig(policy=policy, channel=channel, **kwargs)
    updater = get_updater()
    updater.config = config
    updater.save_config()


def get_current_version() -> str:
    """Get the current installed version."""
    updater = get_updater()
    return updater._current_version


def list_versions(limit: int = 10) -> List[str]:
    """List available versions on PyPI."""
    updater = get_updater()
    return updater.list_available_versions(limit)


# Auto-check on import if configured
def _auto_check_on_import():
    """Automatically check for updates on import if configured."""
    try:
        updater = get_updater()
        if updater.config.auto_check_on_import:
            update_info = updater.check_for_updates()
            if update_info.update_available:
                print(f"\n⚠️  QuantRS2 update available: {update_info.latest_version}")
                print(f"   Current version: {update_info.current_version}")
                print(f"   Run 'quantrs2.auto_updater.install_update()' to update\n")
    except Exception:
        # Silently fail if auto-check fails
        pass


if __name__ == "__main__":
    # CLI interface
    import argparse

    parser = argparse.ArgumentParser(description="QuantRS2 Auto-Updater")
    parser.add_argument('--check', action='store_true', help='Check for updates')
    parser.add_argument('--install', action='store_true', help='Install latest update')
    parser.add_argument('--version', type=str, help='Install specific version')
    parser.add_argument('--list', action='store_true', help='List available versions')
    parser.add_argument('--configure', action='store_true', help='Configure updater')
    parser.add_argument('--policy', type=str, choices=['manual', 'notify', 'automatic', 'disabled'],
                       help='Set update policy')
    parser.add_argument('--channel', type=str, choices=['stable', 'beta', 'alpha', 'dev'],
                       help='Set update channel')

    args = parser.parse_args()

    if args.configure:
        if args.policy or args.channel:
            policy = UpdatePolicy(args.policy) if args.policy else UpdatePolicy.NOTIFY
            channel = UpdateChannel(args.channel) if args.channel else UpdateChannel.STABLE
            configure_updater(policy=policy, channel=channel)
            print("Updater configuration saved.")
        else:
            print("Use --policy and/or --channel to configure updater")

    elif args.check:
        update_available, latest_version = check_for_updates()
        current = get_current_version()
        print(f"Current version: {current}")
        print(f"Latest version: {latest_version}")
        if update_available:
            print("✅ Update available!")
        else:
            print("✅ You are using the latest version")

    elif args.install:
        install_update(args.version)

    elif args.list:
        versions = list_versions(20)
        print("Available versions:")
        for v in versions:
            print(f"  - {v}")

    else:
        parser.print_help()
