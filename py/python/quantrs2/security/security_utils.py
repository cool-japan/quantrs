"""
Security utilities for QuantRS2 production deployment.

This module provides encryption, authentication, and security configuration utilities.
"""

import os
import hashlib
import secrets
import base64
import jwt
import bcrypt
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import logging

logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    """Security configuration settings."""
    
    # Encryption settings
    encryption_algorithm: str = "AES-256-GCM"
    key_derivation_iterations: int = 100000
    salt_length: int = 32
    
    # Authentication settings
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    jwt_refresh_expiration_days: int = 7
    password_min_length: int = 12
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_symbols: bool = True
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst_size: int = 20
    
    # Session security
    session_timeout_minutes: int = 60
    session_cookie_secure: bool = True
    session_cookie_httponly: bool = True
    session_cookie_samesite: str = "strict"
    
    # API security
    api_cors_enabled: bool = True
    api_cors_origins: List[str] = field(default_factory=list)
    api_require_https: bool = True
    api_max_request_size: int = 10 * 1024 * 1024  # 10MB
    
    # Audit settings
    audit_log_enabled: bool = True
    audit_log_path: str = "/var/log/quantrs2/audit.log"
    audit_log_retention_days: int = 365
    
    @classmethod
    def from_env(cls) -> "SecurityConfig":
        """Create security configuration from environment variables."""
        return cls(
            encryption_algorithm=os.getenv("SECURITY_ENCRYPTION_ALGORITHM", "AES-256-GCM"),
            key_derivation_iterations=int(os.getenv("SECURITY_KEY_DERIVATION_ITERATIONS", "100000")),
            salt_length=int(os.getenv("SECURITY_SALT_LENGTH", "32")),
            
            jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            jwt_expiration_hours=int(os.getenv("JWT_EXPIRATION_HOURS", "24")),
            jwt_refresh_expiration_days=int(os.getenv("JWT_REFRESH_EXPIRATION_DAYS", "7")),
            password_min_length=int(os.getenv("PASSWORD_MIN_LENGTH", "12")),
            password_require_uppercase=os.getenv("PASSWORD_REQUIRE_UPPERCASE", "true").lower() == "true",
            password_require_lowercase=os.getenv("PASSWORD_REQUIRE_LOWERCASE", "true").lower() == "true",
            password_require_numbers=os.getenv("PASSWORD_REQUIRE_NUMBERS", "true").lower() == "true",
            password_require_symbols=os.getenv("PASSWORD_REQUIRE_SYMBOLS", "true").lower() == "true",
            
            rate_limit_enabled=os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
            rate_limit_requests_per_minute=int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "100")),
            rate_limit_burst_size=int(os.getenv("RATE_LIMIT_BURST_SIZE", "20")),
            
            session_timeout_minutes=int(os.getenv("SESSION_TIMEOUT_MINUTES", "60")),
            session_cookie_secure=os.getenv("SESSION_COOKIE_SECURE", "true").lower() == "true",
            session_cookie_httponly=os.getenv("SESSION_COOKIE_HTTPONLY", "true").lower() == "true",
            session_cookie_samesite=os.getenv("SESSION_COOKIE_SAMESITE", "strict"),
            
            api_cors_enabled=os.getenv("API_CORS_ENABLED", "true").lower() == "true",
            api_cors_origins=os.getenv("API_CORS_ORIGINS", "").split(",") if os.getenv("API_CORS_ORIGINS") else [],
            api_require_https=os.getenv("API_REQUIRE_HTTPS", "true").lower() == "true",
            api_max_request_size=int(os.getenv("API_MAX_REQUEST_SIZE", str(10 * 1024 * 1024))),
            
            audit_log_enabled=os.getenv("AUDIT_LOG_ENABLED", "true").lower() == "true",
            audit_log_path=os.getenv("AUDIT_LOG_PATH", "/var/log/quantrs2/audit.log"),
            audit_log_retention_days=int(os.getenv("AUDIT_LOG_RETENTION_DAYS", "365")),
        )

def generate_secure_key(length: int = 32) -> str:
    """Generate a cryptographically secure random key."""
    return base64.urlsafe_b64encode(secrets.token_bytes(length)).decode()

def generate_salt(length: int = 32) -> bytes:
    """Generate a cryptographically secure salt."""
    return secrets.token_bytes(length)

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except Exception as e:
        logger.error(f"Password verification failed: {e}")
        return False

def validate_password_strength(password: str, config: SecurityConfig) -> Dict[str, Any]:
    """
    Validate password strength according to security configuration.
    
    Returns:
        Dictionary with validation results
    """
    issues = []
    
    if len(password) < config.password_min_length:
        issues.append(f"Password must be at least {config.password_min_length} characters long")
    
    if config.password_require_uppercase and not any(c.isupper() for c in password):
        issues.append("Password must contain at least one uppercase letter")
    
    if config.password_require_lowercase and not any(c.islower() for c in password):
        issues.append("Password must contain at least one lowercase letter")
    
    if config.password_require_numbers and not any(c.isdigit() for c in password):
        issues.append("Password must contain at least one number")
    
    if config.password_require_symbols and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        issues.append("Password must contain at least one symbol")
    
    # Check for common weak patterns
    weak_patterns = ["123", "abc", "password", "admin", "user", "qwerty"]
    if any(pattern in password.lower() for pattern in weak_patterns):
        issues.append("Password contains common weak patterns")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "strength_score": max(0, 100 - len(issues) * 20)
    }

def encrypt_data(data: Union[str, bytes], key: str) -> str:
    """
    Encrypt data using Fernet (AES 128 in CBC mode).
    
    Args:
        data: Data to encrypt
        key: Encryption key (base64 encoded)
        
    Returns:
        Base64 encoded encrypted data
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    try:
        fernet = Fernet(key.encode() if isinstance(key, str) else key)
        encrypted = fernet.encrypt(data)
        return base64.b64encode(encrypted).decode()
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        raise

def decrypt_data(encrypted_data: str, key: str) -> str:
    """
    Decrypt data using Fernet.
    
    Args:
        encrypted_data: Base64 encoded encrypted data
        key: Decryption key (base64 encoded)
        
    Returns:
        Decrypted data as string
    """
    try:
        fernet = Fernet(key.encode() if isinstance(key, str) else key)
        decoded_data = base64.b64decode(encrypted_data.encode())
        decrypted = fernet.decrypt(decoded_data)
        return decrypted.decode('utf-8')
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        raise

def generate_jwt_token(
    payload: Dict[str, Any], 
    secret_key: str, 
    algorithm: str = "HS256",
    expiration_hours: int = 24
) -> str:
    """
    Generate a JWT token with expiration.
    
    Args:
        payload: Token payload
        secret_key: JWT secret key
        algorithm: JWT algorithm
        expiration_hours: Token expiration in hours
        
    Returns:
        JWT token string
    """
    try:
        # Add standard claims
        now = datetime.utcnow()
        payload.update({
            "iat": now,
            "exp": now + timedelta(hours=expiration_hours),
            "nbf": now,
            "jti": secrets.token_urlsafe(32),  # Unique token ID
        })
        
        token = jwt.encode(payload, secret_key, algorithm=algorithm)
        return token
    except Exception as e:
        logger.error(f"JWT token generation failed: {e}")
        raise

def verify_jwt_token(token: str, secret_key: str, algorithm: str = "HS256") -> Dict[str, Any]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
        secret_key: JWT secret key
        algorithm: JWT algorithm
        
    Returns:
        Decoded token payload
        
    Raises:
        jwt.InvalidTokenError: If token is invalid
    """
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token has expired")
        raise
    except jwt.InvalidTokenError as e:
        logger.error(f"JWT token verification failed: {e}")
        raise

def generate_api_key(prefix: str = "qrs", length: int = 32) -> str:
    """
    Generate a secure API key with prefix.
    
    Args:
        prefix: API key prefix
        length: Length of random part
        
    Returns:
        API key string
    """
    random_part = secrets.token_urlsafe(length)
    return f"{prefix}_{random_part}"

def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()

def verify_api_key(api_key: str, hashed_key: str) -> bool:
    """Verify an API key against its hash."""
    return hashlib.sha256(api_key.encode()).hexdigest() == hashed_key

def generate_csrf_token() -> str:
    """Generate a CSRF token."""
    return secrets.token_urlsafe(32)

def secure_compare(a: str, b: str) -> bool:
    """
    Securely compare two strings to prevent timing attacks.
    
    Args:
        a: First string
        b: Second string
        
    Returns:
        True if strings are equal
    """
    if len(a) != len(b):
        return False
    
    result = 0
    for x, y in zip(a, b):
        result |= ord(x) ^ ord(y)
    
    return result == 0

def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize a filename for safe filesystem usage.
    
    Args:
        filename: Original filename
        max_length: Maximum allowed length
        
    Returns:
        Sanitized filename
    """
    # Remove or replace dangerous characters
    dangerous_chars = '<>:"/\\|?*'
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    # Remove control characters
    filename = ''.join(char for char in filename if ord(char) >= 32)
    
    # Limit length
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        max_name_length = max_length - len(ext)
        filename = name[:max_name_length] + ext
    
    # Ensure filename is not empty
    if not filename or filename.isspace():
        filename = "untitled"
    
    return filename

def get_client_ip(request_headers: Dict[str, str]) -> str:
    """
    Extract client IP address from request headers.
    
    Args:
        request_headers: HTTP request headers
        
    Returns:
        Client IP address
    """
    # Check common proxy headers
    proxy_headers = [
        'X-Forwarded-For',
        'X-Real-IP',
        'X-Client-IP',
        'CF-Connecting-IP',  # Cloudflare
        'X-Cluster-Client-IP',
    ]
    
    for header in proxy_headers:
        if header in request_headers:
            ip = request_headers[header].split(',')[0].strip()
            if ip:
                return ip
    
    # Fallback to remote address
    return request_headers.get('Remote-Addr', 'unknown')

def audit_log_event(
    event_type: str,
    user_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    config: Optional[SecurityConfig] = None
) -> None:
    """
    Log an audit event.
    
    Args:
        event_type: Type of event (login, logout, access, etc.)
        user_id: User identifier
        details: Additional event details
        ip_address: Client IP address
        config: Security configuration
    """
    if not config:
        config = SecurityConfig.from_env()
    
    if not config.audit_log_enabled:
        return
    
    try:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "ip_address": ip_address,
            "details": details or {},
        }
        
        # Ensure log directory exists
        log_path = Path(config.audit_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write log entry
        with open(log_path, 'a') as f:
            f.write(f"{log_entry}\n")
            
    except Exception as e:
        logger.error(f"Failed to write audit log: {e}")

class SecurityHeaders:
    """Security headers for HTTP responses."""
    
    @staticmethod
    def get_security_headers(config: SecurityConfig) -> Dict[str, str]:
        """Get recommended security headers."""
        headers = {
            # Content Security Policy
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' https:; "
                "connect-src 'self' https:; "
                "frame-ancestors 'none'"
            ),
            
            # XSS Protection
            "X-XSS-Protection": "1; mode=block",
            
            # Content Type Options
            "X-Content-Type-Options": "nosniff",
            
            # Frame Options
            "X-Frame-Options": "DENY",
            
            # HSTS (if HTTPS required)
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload" if config.api_require_https else None,
            
            # Referrer Policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Permissions Policy
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            
            # Remove server information
            "Server": "QuantRS2",
        }
        
        # Remove None values
        return {k: v for k, v in headers.items() if v is not None}