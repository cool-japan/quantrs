//! Credential management for quantum cloud providers.
//!
//! Abstracts credential storage behind a trait so the same code works
//! with environment variables, files, or future vault integrations.

use std::collections::HashMap;

/// A secret string that zeroes its memory on drop to reduce secret leakage.
///
/// Does not implement `Clone` to prevent accidental copies. Access the
/// underlying value with [`SecretString::expose_secret`] in a short-lived context.
pub struct SecretString {
    inner: Vec<u8>,
}

impl SecretString {
    /// Wrap a string value as a secret
    pub fn new(s: impl Into<String>) -> Self {
        Self {
            inner: s.into().into_bytes(),
        }
    }

    /// Access the secret value as a string slice.
    ///
    /// Keep the borrow as short-lived as possible to limit exposure.
    pub fn expose_secret(&self) -> &str {
        // SAFETY: We stored valid UTF-8 from Into<String>.
        std::str::from_utf8(&self.inner).unwrap_or("")
    }

    /// Return the length in bytes
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Return true if the secret is empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl Drop for SecretString {
    fn drop(&mut self) {
        // Volatile write prevents the compiler from optimizing away the zeroing.
        for b in self.inner.iter_mut() {
            // SAFETY: `b` is a valid mutable reference to a byte within our Vec.
            unsafe {
                std::ptr::write_volatile(b, 0u8);
            }
        }
    }
}

impl std::fmt::Debug for SecretString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[REDACTED]")
    }
}

impl std::fmt::Display for SecretString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[REDACTED]")
    }
}

/// Error type for credential operations
#[derive(Debug)]
#[non_exhaustive]
pub enum CredentialError {
    /// The requested credential key was not found
    NotFound(String),
    /// File permissions are too permissive
    PermissionDenied(String),
    /// Underlying I/O failure
    IoError(std::io::Error),
    /// Failed to parse credential file
    ParseError(String),
}

impl std::fmt::Display for CredentialError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CredentialError::NotFound(k) => write!(f, "credential not found: {}", k),
            CredentialError::PermissionDenied(p) => write!(f, "permission denied: {}", p),
            CredentialError::IoError(e) => write!(f, "credential IO error: {}", e),
            CredentialError::ParseError(s) => write!(f, "credential parse error: {}", s),
        }
    }
}

impl std::error::Error for CredentialError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CredentialError::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for CredentialError {
    fn from(e: std::io::Error) -> Self {
        CredentialError::IoError(e)
    }
}

/// Abstraction for credential storage backends
pub trait CredentialProvider: Send + Sync {
    /// Retrieve a credential by key, returning a zeroing `SecretString`
    fn get_credential(&self, key: &str) -> Result<SecretString, CredentialError>;

    /// List available credential keys.
    ///
    /// May return an empty vec for security reasons (e.g. env-var providers
    /// avoid enumerating the process environment).
    fn available_keys(&self) -> Vec<String>;
}

/// Reads credentials from environment variables.
///
/// Keys are looked up as `{PREFIX}_{KEY}` (both uppercased). If the prefix
/// is empty the key is used directly, uppercased.
///
/// # Example
///
/// ```rust,no_run
/// # use quantrs2_device::security::credentials::{EnvVarCredentialProvider, CredentialProvider};
/// let provider = EnvVarCredentialProvider::new("QUANTRS");
/// // looks for env var "QUANTRS_IBM_TOKEN"
/// let _ = provider.get_credential("IBM_TOKEN");
/// ```
pub struct EnvVarCredentialProvider {
    prefix: String,
}

impl EnvVarCredentialProvider {
    /// Create a new provider with the given environment variable prefix
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into().to_uppercase(),
        }
    }

    fn env_key(&self, key: &str) -> String {
        if self.prefix.is_empty() {
            key.to_uppercase()
        } else {
            format!("{}_{}", self.prefix, key.to_uppercase())
        }
    }
}

impl CredentialProvider for EnvVarCredentialProvider {
    fn get_credential(&self, key: &str) -> Result<SecretString, CredentialError> {
        let env_key = self.env_key(key);
        std::env::var(&env_key)
            .map(SecretString::new)
            .map_err(|_| CredentialError::NotFound(env_key))
    }

    fn available_keys(&self) -> Vec<String> {
        // Don't enumerate environment variables — it's a security risk
        vec![]
    }
}

/// Reads credentials from a JSON file (e.g. `~/.quantrs/credentials.json`).
///
/// The file must contain a flat JSON object mapping string keys to string values.
/// On Unix systems the file must have mode `0o600`; looser permissions are rejected.
///
/// # Example file
///
/// ```json
/// {
///   "IBM_TOKEN": "my-ibm-token",
///   "AWS_SECRET": "my-aws-secret"
/// }
/// ```
pub struct FileCredentialProvider {
    path: std::path::PathBuf,
}

impl FileCredentialProvider {
    /// Create a provider that reads from the given JSON credentials file
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self { path: path.into() }
    }

    fn load(&self) -> Result<HashMap<String, String>, CredentialError> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let meta = std::fs::metadata(&self.path).map_err(CredentialError::IoError)?;
            if meta.mode() & 0o077 != 0 {
                return Err(CredentialError::PermissionDenied(format!(
                    "credentials file {:?} must have mode 0600",
                    self.path
                )));
            }
        }

        let content = std::fs::read_to_string(&self.path).map_err(CredentialError::IoError)?;
        serde_json::from_str::<HashMap<String, String>>(&content)
            .map_err(|e| CredentialError::ParseError(e.to_string()))
    }
}

impl CredentialProvider for FileCredentialProvider {
    fn get_credential(&self, key: &str) -> Result<SecretString, CredentialError> {
        let map = self.load()?;
        map.get(key)
            .map(|v| SecretString::new(v.clone()))
            .ok_or_else(|| CredentialError::NotFound(key.to_string()))
    }

    fn available_keys(&self) -> Vec<String> {
        self.load()
            .map(|m| m.into_keys().collect())
            .unwrap_or_default()
    }
}

/// Tries multiple [`CredentialProvider`]s in order until one succeeds.
///
/// Useful for layering: environment variables override file credentials, which
/// override compiled-in defaults.
///
/// # Example
///
/// ```rust,no_run
/// # use quantrs2_device::security::credentials::{
/// #     CompositeCredentialProvider, EnvVarCredentialProvider, CredentialProvider
/// # };
/// let provider = CompositeCredentialProvider::new()
///     .with_provider(EnvVarCredentialProvider::new("QUANTRS"));
/// let _ = provider.get_credential("IBM_TOKEN");
/// ```
pub struct CompositeCredentialProvider {
    providers: Vec<Box<dyn CredentialProvider>>,
}

impl CompositeCredentialProvider {
    /// Create an empty composite provider (no sources yet)
    pub fn new() -> Self {
        Self { providers: vec![] }
    }

    /// Append a credential provider source (tried in insertion order)
    pub fn with_provider(mut self, p: impl CredentialProvider + 'static) -> Self {
        self.providers.push(Box::new(p));
        self
    }
}

impl Default for CompositeCredentialProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl CredentialProvider for CompositeCredentialProvider {
    fn get_credential(&self, key: &str) -> Result<SecretString, CredentialError> {
        for p in &self.providers {
            if let Ok(s) = p.get_credential(key) {
                return Ok(s);
            }
        }
        Err(CredentialError::NotFound(key.to_string()))
    }

    fn available_keys(&self) -> Vec<String> {
        self.providers
            .iter()
            .flat_map(|p| p.available_keys())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::io::Write;

    #[test]
    fn test_secret_string_expose() {
        let secret = SecretString::new("my-api-key");
        assert_eq!(secret.expose_secret(), "my-api-key");
    }

    #[test]
    fn test_secret_string_debug_redacted() {
        let secret = SecretString::new("super-secret");
        assert_eq!(format!("{:?}", secret), "[REDACTED]");
        assert_eq!(format!("{}", secret), "[REDACTED]");
    }

    #[test]
    fn test_secret_string_len() {
        let secret = SecretString::new("hello");
        assert_eq!(secret.len(), 5);
        assert!(!secret.is_empty());
    }

    #[test]
    fn test_secret_string_empty() {
        let secret = SecretString::new("");
        assert!(secret.is_empty());
        assert_eq!(secret.len(), 0);
    }

    #[test]
    fn test_env_var_provider_found() {
        let key = format!("QUANTRS_TEST_KEY_{}", fastrand::u64(..));
        env::set_var(&key, "test-token");

        let provider = EnvVarCredentialProvider::new("");
        let secret = provider.get_credential(&key).expect("should find env var");
        assert_eq!(secret.expose_secret(), "test-token");

        env::remove_var(&key);
    }

    #[test]
    fn test_env_var_provider_with_prefix() {
        let suffix = fastrand::u64(..);
        let env_var = format!("QUANTRS_TEST_{}", suffix);
        env::set_var(&env_var, "prefixed-value");

        let provider = EnvVarCredentialProvider::new("QUANTRS");
        let secret = provider
            .get_credential(&format!("TEST_{}", suffix))
            .expect("should find prefixed env var");
        assert_eq!(secret.expose_secret(), "prefixed-value");

        env::remove_var(&env_var);
    }

    #[test]
    fn test_env_var_provider_not_found() {
        let provider = EnvVarCredentialProvider::new("QUANTRS");
        let result = provider.get_credential("DEFINITELY_NONEXISTENT_KEY_12345");
        assert!(matches!(result, Err(CredentialError::NotFound(_))));
    }

    #[test]
    fn test_env_var_provider_available_keys_empty() {
        let provider = EnvVarCredentialProvider::new("QUANTRS");
        assert!(provider.available_keys().is_empty());
    }

    #[test]
    fn test_file_credential_provider() {
        let dir = env::temp_dir();
        let path = dir.join(format!("quantrs_creds_{}.json", fastrand::u64(..)));

        let mut file = std::fs::File::create(&path).expect("create file");
        write!(file, r#"{{"IBM_TOKEN":"ibm-secret","AWS_KEY":"aws-secret"}}"#)
            .expect("write credentials");
        drop(file);

        // Set mode 0600 on Unix so the permission check passes
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            std::fs::set_permissions(&path, perms).expect("set permissions");
        }

        let provider = FileCredentialProvider::new(&path);
        let secret = provider
            .get_credential("IBM_TOKEN")
            .expect("should find IBM_TOKEN");
        assert_eq!(secret.expose_secret(), "ibm-secret");

        let keys = provider.available_keys();
        assert!(keys.contains(&"IBM_TOKEN".to_string()) || keys.contains(&"AWS_KEY".to_string()));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_file_credential_provider_not_found() {
        let dir = env::temp_dir();
        let path = dir.join(format!("quantrs_creds_nf_{}.json", fastrand::u64(..)));

        let mut file = std::fs::File::create(&path).expect("create file");
        write!(file, r#"{{"IBM_TOKEN":"ibm-secret"}}"#).expect("write credentials");
        drop(file);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            std::fs::set_permissions(&path, perms).expect("set permissions");
        }

        let provider = FileCredentialProvider::new(&path);
        let result = provider.get_credential("NONEXISTENT");
        assert!(matches!(result, Err(CredentialError::NotFound(_))));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_composite_provider_fallthrough() {
        // First provider has nothing; second has the key
        let suffix = fastrand::u64(..);
        let env_var = format!("QUANTRS_COMPOSITE_{}", suffix);
        env::set_var(&env_var, "composite-value");

        // Env provider with wrong prefix → won't find it
        let missing_provider = EnvVarCredentialProvider::new("WRONG_PREFIX");
        // Env provider with no prefix → will find it as-is
        let found_provider = EnvVarCredentialProvider::new("");

        let composite = CompositeCredentialProvider::new()
            .with_provider(missing_provider)
            .with_provider(found_provider);

        let secret = composite
            .get_credential(&env_var)
            .expect("composite should find via second provider");
        assert_eq!(secret.expose_secret(), "composite-value");

        env::remove_var(&env_var);
    }

    #[test]
    fn test_composite_provider_all_fail() {
        let composite = CompositeCredentialProvider::new()
            .with_provider(EnvVarCredentialProvider::new("DEFINITELY_MISSING_PREFIX_XYZ"));

        let result = composite.get_credential("NONEXISTENT");
        assert!(matches!(result, Err(CredentialError::NotFound(_))));
    }

    #[test]
    fn test_composite_provider_empty() {
        let composite = CompositeCredentialProvider::new();
        let result = composite.get_credential("any_key");
        assert!(matches!(result, Err(CredentialError::NotFound(_))));
        assert!(composite.available_keys().is_empty());
    }

    #[test]
    fn test_credential_error_display() {
        let e = CredentialError::NotFound("MY_KEY".to_string());
        assert!(e.to_string().contains("MY_KEY"));

        let e = CredentialError::PermissionDenied("/path/to/file".to_string());
        assert!(e.to_string().contains("permission denied"));

        let e = CredentialError::ParseError("invalid json".to_string());
        assert!(e.to_string().contains("parse error"));
    }
}
