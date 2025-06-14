"""
Authentication and Authorization Management for QuantRS2

This module provides comprehensive user authentication, authorization,
and access control for production deployment.
"""

import os
import time
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import jwt
import bcrypt
from pathlib import Path
import logging

from .security_utils import (
    SecurityConfig, hash_password, verify_password, generate_jwt_token,
    verify_jwt_token, audit_log_event, secure_compare
)

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles with hierarchical permissions."""
    ADMIN = "admin"
    RESEARCHER = "researcher"
    DEVELOPER = "developer"
    USER = "user"
    READONLY = "readonly"

class Permission(Enum):
    """System permissions."""
    # System administration
    ADMIN_SYSTEM = "admin:system"
    ADMIN_USERS = "admin:users"
    ADMIN_QUANTUM_BACKENDS = "admin:quantum_backends"
    
    # Quantum computing
    QUANTUM_EXECUTE = "quantum:execute"
    QUANTUM_ADVANCED = "quantum:advanced"
    QUANTUM_HARDWARE = "quantum:hardware"
    QUANTUM_CLOUD = "quantum:cloud"
    
    # Data management
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"
    DATA_EXPORT = "data:export"
    
    # Development tools
    DEV_DEBUG = "dev:debug"
    DEV_PROFILING = "dev:profiling"
    DEV_TESTING = "dev:testing"
    
    # API access
    API_READ = "api:read"
    API_WRITE = "api:write"
    API_ADMIN = "api:admin"

@dataclass
class User:
    """User account information."""
    
    id: str
    username: str
    email: str
    password_hash: str
    role: UserRole
    permissions: Set[Permission] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    is_active: bool = True
    is_verified: bool = False
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    api_keys: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary for storage."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "password_hash": self.password_hash,
            "role": self.role.value,
            "permissions": [p.value for p in self.permissions],
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "failed_login_attempts": self.failed_login_attempts,
            "locked_until": self.locked_until.isoformat() if self.locked_until else None,
            "api_keys": self.api_keys,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        """Create user from dictionary."""
        return cls(
            id=data["id"],
            username=data["username"],
            email=data["email"],
            password_hash=data["password_hash"],
            role=UserRole(data["role"]),
            permissions=set(Permission(p) for p in data.get("permissions", [])),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_login=datetime.fromisoformat(data["last_login"]) if data.get("last_login") else None,
            is_active=data.get("is_active", True),
            is_verified=data.get("is_verified", False),
            failed_login_attempts=data.get("failed_login_attempts", 0),
            locked_until=datetime.fromisoformat(data["locked_until"]) if data.get("locked_until") else None,
            api_keys=data.get("api_keys", []),
            metadata=data.get("metadata", {}),
        )

@dataclass
class Session:
    """User session information."""
    
    id: str
    user_id: str
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class AuthenticationManager:
    """
    Comprehensive authentication manager.
    
    Features:
    - Password-based authentication
    - JWT token management
    - API key authentication
    - Session management
    - Account lockout protection
    - Multi-factor authentication support
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig.from_env()
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self._load_users()
        self._cleanup_expired_sessions()
    
    def _load_users(self) -> None:
        """Load users from storage."""
        # In production, this would load from a database
        # For now, create a default admin user if none exist
        if not self.users:
            admin_user = self.create_user(
                username="admin",
                email="admin@quantrs2.local",
                password="ChangeMe123!",
                role=UserRole.ADMIN,
            )
            logger.warning("Created default admin user. Change password immediately!")
    
    def _save_users(self) -> None:
        """Save users to storage."""
        # In production, this would save to a database
        pass
    
    def _cleanup_expired_sessions(self) -> None:
        """Remove expired sessions."""
        now = datetime.utcnow()
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session.expires_at < now
        ]
        for session_id in expired_sessions:
            del self.sessions[session_id]
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.USER,
        permissions: Optional[Set[Permission]] = None,
    ) -> User:
        """
        Create a new user account.
        
        Args:
            username: Unique username
            email: User email address
            password: Plain text password
            role: User role
            permissions: Additional permissions
            
        Returns:
            Created user object
            
        Raises:
            ValueError: If user creation fails
        """
        # Validate inputs
        if not username or len(username) < 3:
            raise ValueError("Username must be at least 3 characters")
        
        if not email or "@" not in email:
            raise ValueError("Valid email address required")
        
        # Check if user already exists
        for user in self.users.values():
            if user.username == username:
                raise ValueError("Username already exists")
            if user.email == email:
                raise ValueError("Email already exists")
        
        # Validate password strength
        from .security_utils import validate_password_strength
        password_validation = validate_password_strength(password, self.config)
        if not password_validation["valid"]:
            raise ValueError(f"Password validation failed: {password_validation['issues']}")
        
        # Create user
        user_id = secrets.token_urlsafe(16)
        password_hash = hash_password(password)
        
        # Set default permissions based on role
        if permissions is None:
            permissions = self._get_default_permissions(role)
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            role=role,
            permissions=permissions,
        )
        
        self.users[user_id] = user
        self._save_users()
        
        audit_log_event(
            "user_created",
            user_id=user_id,
            details={"username": username, "email": email, "role": role.value},
            config=self.config,
        )
        
        logger.info(f"Created user: {username} ({user_id})")
        return user
    
    def _get_default_permissions(self, role: UserRole) -> Set[Permission]:
        """Get default permissions for a role."""
        permission_map = {
            UserRole.ADMIN: {
                Permission.ADMIN_SYSTEM,
                Permission.ADMIN_USERS,
                Permission.ADMIN_QUANTUM_BACKENDS,
                Permission.QUANTUM_EXECUTE,
                Permission.QUANTUM_ADVANCED,
                Permission.QUANTUM_HARDWARE,
                Permission.QUANTUM_CLOUD,
                Permission.DATA_READ,
                Permission.DATA_WRITE,
                Permission.DATA_DELETE,
                Permission.DATA_EXPORT,
                Permission.DEV_DEBUG,
                Permission.DEV_PROFILING,
                Permission.DEV_TESTING,
                Permission.API_READ,
                Permission.API_WRITE,
                Permission.API_ADMIN,
            },
            UserRole.RESEARCHER: {
                Permission.QUANTUM_EXECUTE,
                Permission.QUANTUM_ADVANCED,
                Permission.QUANTUM_CLOUD,
                Permission.DATA_READ,
                Permission.DATA_WRITE,
                Permission.DATA_EXPORT,
                Permission.DEV_PROFILING,
                Permission.API_READ,
                Permission.API_WRITE,
            },
            UserRole.DEVELOPER: {
                Permission.QUANTUM_EXECUTE,
                Permission.QUANTUM_ADVANCED,
                Permission.DATA_READ,
                Permission.DATA_WRITE,
                Permission.DEV_DEBUG,
                Permission.DEV_PROFILING,
                Permission.DEV_TESTING,
                Permission.API_READ,
                Permission.API_WRITE,
            },
            UserRole.USER: {
                Permission.QUANTUM_EXECUTE,
                Permission.DATA_READ,
                Permission.API_READ,
            },
            UserRole.READONLY: {
                Permission.DATA_READ,
                Permission.API_READ,
            },
        }
        
        return permission_map.get(role, set())
    
    def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: str = "unknown",
        user_agent: str = "unknown",
    ) -> Optional[str]:
        """
        Authenticate a user and create a session.
        
        Args:
            username: Username or email
            password: Plain text password
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Session token if authentication successful, None otherwise
        """
        user = self._find_user_by_username_or_email(username)
        if not user:
            audit_log_event(
                "authentication_failed",
                details={"username": username, "reason": "user_not_found"},
                ip_address=ip_address,
                config=self.config,
            )
            return None
        
        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.utcnow():
            audit_log_event(
                "authentication_failed",
                user_id=user.id,
                details={"username": username, "reason": "account_locked"},
                ip_address=ip_address,
                config=self.config,
            )
            return None
        
        # Check if account is active
        if not user.is_active:
            audit_log_event(
                "authentication_failed",
                user_id=user.id,
                details={"username": username, "reason": "account_inactive"},
                ip_address=ip_address,
                config=self.config,
            )
            return None
        
        # Verify password
        if not verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            
            # Lock account after 5 failed attempts
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.utcnow() + timedelta(minutes=30)
                logger.warning(f"Account locked due to failed login attempts: {user.username}")
            
            self._save_users()
            
            audit_log_event(
                "authentication_failed",
                user_id=user.id,
                details={
                    "username": username,
                    "reason": "invalid_password",
                    "failed_attempts": user.failed_login_attempts,
                },
                ip_address=ip_address,
                config=self.config,
            )
            return None
        
        # Authentication successful
        user.failed_login_attempts = 0
        user.last_login = datetime.utcnow()
        user.locked_until = None
        self._save_users()
        
        # Create session
        session_id = self._create_session(user.id, ip_address, user_agent)
        
        audit_log_event(
            "authentication_successful",
            user_id=user.id,
            details={"username": username, "session_id": session_id},
            ip_address=ip_address,
            config=self.config,
        )
        
        logger.info(f"User authenticated: {user.username} ({user.id})")
        return session_id
    
    def _find_user_by_username_or_email(self, identifier: str) -> Optional[User]:
        """Find user by username or email."""
        for user in self.users.values():
            if user.username == identifier or user.email == identifier:
                return user
        return None
    
    def _create_session(self, user_id: str, ip_address: str, user_agent: str) -> str:
        """Create a new user session."""
        session_id = secrets.token_urlsafe(32)
        now = datetime.utcnow()
        expires_at = now + timedelta(minutes=self.config.session_timeout_minutes)
        
        session = Session(
            id=session_id,
            user_id=user_id,
            created_at=now,
            last_accessed=now,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        
        self.sessions[session_id] = session
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[User]:
        """
        Validate a session and return the associated user.
        
        Args:
            session_id: Session identifier
            
        Returns:
            User object if session is valid, None otherwise
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        now = datetime.utcnow()
        
        # Check if session is expired
        if session.expires_at < now:
            del self.sessions[session_id]
            return None
        
        # Check if session is active
        if not session.is_active:
            return None
        
        # Update last accessed time
        session.last_accessed = now
        session.expires_at = now + timedelta(minutes=self.config.session_timeout_minutes)
        
        # Get user
        user = self.users.get(session.user_id)
        if not user or not user.is_active:
            del self.sessions[session_id]
            return None
        
        return user
    
    def logout_user(self, session_id: str) -> bool:
        """
        Logout a user by invalidating their session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if logout successful
        """
        if session_id in self.sessions:
            session = self.sessions[session_id]
            user_id = session.user_id
            
            del self.sessions[session_id]
            
            audit_log_event(
                "user_logout",
                user_id=user_id,
                details={"session_id": session_id},
                config=self.config,
            )
            
            return True
        
        return False
    
    def generate_api_key(self, user_id: str, name: str = "default") -> str:
        """
        Generate an API key for a user.
        
        Args:
            user_id: User identifier
            name: API key name/description
            
        Returns:
            API key string
        """
        user = self.users.get(user_id)
        if not user:
            raise ValueError("User not found")
        
        api_key = f"qrs_{secrets.token_urlsafe(32)}"
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Store hashed API key
        user.api_keys.append({
            "hash": api_key_hash,
            "name": name,
            "created_at": datetime.utcnow().isoformat(),
        })
        
        self._save_users()
        
        audit_log_event(
            "api_key_generated",
            user_id=user_id,
            details={"name": name},
            config=self.config,
        )
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[User]:
        """
        Validate an API key and return the associated user.
        
        Args:
            api_key: API key to validate
            
        Returns:
            User object if API key is valid, None otherwise
        """
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        for user in self.users.values():
            if not user.is_active:
                continue
            
            for key_info in user.api_keys:
                if secure_compare(key_info["hash"], api_key_hash):
                    return user
        
        return None

class AuthorizationManager:
    """
    Authorization and access control manager.
    
    Features:
    - Role-based access control (RBAC)
    - Permission-based authorization
    - Resource-level access control
    - Quantum backend access control
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig.from_env()
    
    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        return permission in user.permissions
    
    def has_role(self, user: User, role: UserRole) -> bool:
        """Check if user has a specific role."""
        return user.role == role
    
    def can_access_quantum_backend(self, user: User, backend_type: str) -> bool:
        """Check if user can access a quantum backend type."""
        if backend_type == "simulator":
            return self.has_permission(user, Permission.QUANTUM_EXECUTE)
        elif backend_type == "hardware":
            return self.has_permission(user, Permission.QUANTUM_HARDWARE)
        elif backend_type == "cloud":
            return self.has_permission(user, Permission.QUANTUM_CLOUD)
        else:
            return False
    
    def can_execute_quantum_circuit(
        self,
        user: User,
        circuit_complexity: str = "basic"
    ) -> bool:
        """Check if user can execute a quantum circuit based on complexity."""
        if circuit_complexity == "basic":
            return self.has_permission(user, Permission.QUANTUM_EXECUTE)
        elif circuit_complexity == "advanced":
            return self.has_permission(user, Permission.QUANTUM_ADVANCED)
        else:
            return False
    
    def can_access_data(self, user: User, operation: str) -> bool:
        """Check if user can perform data operations."""
        operation_map = {
            "read": Permission.DATA_READ,
            "write": Permission.DATA_WRITE,
            "delete": Permission.DATA_DELETE,
            "export": Permission.DATA_EXPORT,
        }
        
        required_permission = operation_map.get(operation)
        if not required_permission:
            return False
        
        return self.has_permission(user, required_permission)
    
    def can_use_development_tools(self, user: User, tool: str) -> bool:
        """Check if user can use development tools."""
        tool_map = {
            "debugger": Permission.DEV_DEBUG,
            "profiler": Permission.DEV_PROFILING,
            "testing": Permission.DEV_TESTING,
        }
        
        required_permission = tool_map.get(tool)
        if not required_permission:
            return False
        
        return self.has_permission(user, required_permission)
    
    def filter_accessible_resources(
        self,
        user: User,
        resources: List[Dict[str, Any]],
        resource_type: str = "quantum_circuit"
    ) -> List[Dict[str, Any]]:
        """Filter resources based on user access rights."""
        accessible_resources = []
        
        for resource in resources:
            if self._can_access_resource(user, resource, resource_type):
                accessible_resources.append(resource)
        
        return accessible_resources
    
    def _can_access_resource(
        self,
        user: User,
        resource: Dict[str, Any],
        resource_type: str
    ) -> bool:
        """Check if user can access a specific resource."""
        # Owner can always access their resources
        if resource.get("owner_id") == user.id:
            return True
        
        # Check if resource is public
        if resource.get("is_public", False):
            return self.has_permission(user, Permission.DATA_READ)
        
        # Check if user has admin permissions
        if self.has_permission(user, Permission.ADMIN_SYSTEM):
            return True
        
        # Check resource-specific permissions
        if resource_type == "quantum_circuit":
            return self.has_permission(user, Permission.QUANTUM_EXECUTE)
        elif resource_type == "quantum_result":
            return self.has_permission(user, Permission.DATA_READ)
        
        return False

# Global instances
_auth_manager: Optional[AuthenticationManager] = None
_authz_manager: Optional[AuthorizationManager] = None

def get_auth_manager() -> AuthenticationManager:
    """Get global authentication manager instance."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthenticationManager()
    return _auth_manager

def get_authz_manager() -> AuthorizationManager:
    """Get global authorization manager instance."""
    global _authz_manager
    if _authz_manager is None:
        _authz_manager = AuthorizationManager()
    return _authz_manager