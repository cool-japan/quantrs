//! Session and User Management for Performance Dashboard
//!
//! This module handles user sessions, permissions, preferences, and access control
//! for the performance analytics dashboard.

use crate::DeviceResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// User session management
pub struct SessionManager {
    active_sessions: HashMap<String, UserSession>,
    session_config: SessionConfig,
    auth_provider: Box<dyn AuthProvider + Send + Sync>,
    permission_manager: PermissionManager,
}

/// User session information
#[derive(Debug, Clone)]
pub struct UserSession {
    pub session_id: String,
    pub user_id: String,
    pub permissions: Vec<Permission>,
    pub preferences: UserPreferences,
    pub last_activity: SystemTime,
    pub session_data: HashMap<String, String>,
    pub auth_token: Option<String>,
    pub expires_at: SystemTime,
}

/// Session configuration
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub session_timeout: Duration,
    pub max_concurrent_sessions: usize,
    pub require_authentication: bool,
    pub enable_session_persistence: bool,
    pub cookie_settings: CookieSettings,
    pub security_settings: SecuritySettings,
}

/// Cookie settings
#[derive(Debug, Clone)]
pub struct CookieSettings {
    pub secure: bool,
    pub http_only: bool,
    pub same_site: SameSitePolicy,
    pub domain: Option<String>,
    pub path: String,
    pub max_age: Duration,
}

/// Same-site policy
#[derive(Debug, Clone, PartialEq)]
pub enum SameSitePolicy {
    Strict,
    Lax,
    None,
}

/// Security settings
#[derive(Debug, Clone)]
pub struct SecuritySettings {
    pub csrf_protection: bool,
    pub rate_limiting: bool,
    pub ip_whitelist: Option<Vec<String>>,
    pub require_https: bool,
    pub enable_audit_logging: bool,
}

/// User permissions
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum Permission {
    ReadDashboard,
    WriteDashboard,
    ManageAlerts,
    ExportData,
    ViewReports,
    ManageUsers,
    SystemAdmin,
    Custom(String),
}

/// User preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    pub dashboard_layout: String,
    pub default_time_range: String,
    pub chart_preferences: ChartPreferences,
    pub notification_preferences: NotificationPreferences,
    pub display_preferences: DisplayPreferences,
    pub custom_preferences: HashMap<String, String>,
}

/// Chart preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartPreferences {
    pub preferred_chart_types: Vec<String>,
    pub color_scheme: String,
    pub animation_enabled: bool,
    pub interactive_features: bool,
    pub default_aggregation: String,
    pub refresh_interval: u64,
}

/// Notification preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationPreferences {
    pub email_notifications: bool,
    pub browser_notifications: bool,
    pub slack_notifications: bool,
    pub notification_frequency: NotificationFrequency,
    pub alert_thresholds: HashMap<String, f64>,
    pub quiet_hours: Option<QuietHours>,
}

/// Notification frequency
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NotificationFrequency {
    Immediate,
    Hourly,
    Daily,
    Weekly,
    Custom(Duration),
}

/// Quiet hours configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuietHours {
    pub start_time: String, // HH:MM format
    pub end_time: String,   // HH:MM format
    pub timezone: String,
    pub days_of_week: Vec<DayOfWeek>,
}

/// Days of the week
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DayOfWeek {
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
    Sunday,
}

/// Display preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayPreferences {
    pub theme: String,
    pub font_size: String,
    pub density: DisplayDensity,
    pub sidebar_collapsed: bool,
    pub show_tooltips: bool,
    pub language: String,
    pub timezone: String,
}

/// Display density
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DisplayDensity {
    Compact,
    Normal,
    Spacious,
}

/// Authentication provider trait
pub trait AuthProvider {
    fn authenticate(&self, credentials: &Credentials) -> DeviceResult<AuthResult>;
    fn validate_token(&self, token: &str) -> DeviceResult<TokenValidation>;
    fn refresh_token(&self, refresh_token: &str) -> DeviceResult<AuthResult>;
    fn logout(&self, token: &str) -> DeviceResult<()>;
}

/// Authentication credentials
#[derive(Debug, Clone)]
pub struct Credentials {
    pub username: Option<String>,
    pub password: Option<String>,
    pub token: Option<String>,
    pub oauth_code: Option<String>,
    pub provider: AuthProviderType,
}

/// Authentication provider types
#[derive(Debug, Clone, PartialEq)]
pub enum AuthProviderType {
    Local,
    LDAP,
    OAuth2,
    SAML,
    JWT,
    Custom(String),
}

/// Authentication result
#[derive(Debug, Clone)]
pub struct AuthResult {
    pub success: bool,
    pub user_info: Option<UserInfo>,
    pub access_token: Option<String>,
    pub refresh_token: Option<String>,
    pub expires_at: Option<SystemTime>,
    pub error_message: Option<String>,
}

/// User information
#[derive(Debug, Clone)]
pub struct UserInfo {
    pub user_id: String,
    pub username: String,
    pub email: String,
    pub full_name: String,
    pub roles: Vec<Role>,
    pub groups: Vec<String>,
    pub metadata: HashMap<String, String>,
}

/// User role
#[derive(Debug, Clone)]
pub struct Role {
    pub role_id: String,
    pub role_name: String,
    pub permissions: Vec<Permission>,
    pub description: String,
}

/// Token validation result
#[derive(Debug, Clone)]
pub struct TokenValidation {
    pub valid: bool,
    pub user_id: Option<String>,
    pub expires_at: Option<SystemTime>,
    pub scopes: Vec<String>,
}

/// Permission manager for access control
pub struct PermissionManager {
    role_definitions: HashMap<String, Role>,
    permission_cache: HashMap<String, Vec<Permission>>,
    access_policies: Vec<AccessPolicy>,
}

/// Access policy
#[derive(Debug, Clone)]
pub struct AccessPolicy {
    pub policy_id: String,
    pub resource: String,
    pub action: String,
    pub conditions: Vec<AccessCondition>,
    pub effect: PolicyEffect,
}

/// Policy effect
#[derive(Debug, Clone, PartialEq)]
pub enum PolicyEffect {
    Allow,
    Deny,
}

/// Access condition
#[derive(Debug, Clone)]
pub struct AccessCondition {
    pub condition_type: ConditionType,
    pub operator: ConditionOperator,
    pub value: String,
}

/// Condition types
#[derive(Debug, Clone, PartialEq)]
pub enum ConditionType {
    UserRole,
    UserGroup,
    TimeOfDay,
    DayOfWeek,
    IpAddress,
    ResourceOwner,
    Custom(String),
}

/// Condition operators
#[derive(Debug, Clone, PartialEq)]
pub enum ConditionOperator {
    Equals,
    NotEquals,
    Contains,
    StartsWith,
    EndsWith,
    GreaterThan,
    LessThan,
    InRange,
    Custom(String),
}

/// Session activity tracking
#[derive(Debug, Clone)]
pub struct SessionActivity {
    pub session_id: String,
    pub user_id: String,
    pub activity_type: ActivityType,
    pub timestamp: SystemTime,
    pub ip_address: String,
    pub user_agent: String,
    pub resource_accessed: Option<String>,
    pub metadata: HashMap<String, String>,
}

/// Activity types
#[derive(Debug, Clone, PartialEq)]
pub enum ActivityType {
    Login,
    Logout,
    DashboardView,
    ChartInteraction,
    DataExport,
    ConfigurationChange,
    AlertAcknowledgement,
    Custom(String),
}

/// Audit logging
pub struct AuditLogger {
    log_config: AuditLogConfig,
    log_storage: Box<dyn LogStorage + Send + Sync>,
    log_buffer: Vec<AuditEvent>,
}

/// Audit log configuration
#[derive(Debug, Clone)]
pub struct AuditLogConfig {
    pub enabled: bool,
    pub log_level: AuditLogLevel,
    pub retention_period: Duration,
    pub include_sensitive_data: bool,
    pub encryption_enabled: bool,
}

/// Audit log levels
#[derive(Debug, Clone, PartialEq)]
pub enum AuditLogLevel {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

/// Audit event
#[derive(Debug, Clone)]
pub struct AuditEvent {
    pub event_id: String,
    pub timestamp: SystemTime,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub event_type: AuditEventType,
    pub resource: String,
    pub action: String,
    pub result: AuditResult,
    pub ip_address: String,
    pub user_agent: String,
    pub details: HashMap<String, String>,
}

/// Audit event types
#[derive(Debug, Clone, PartialEq)]
pub enum AuditEventType {
    Authentication,
    Authorization,
    DataAccess,
    DataModification,
    SystemChange,
    SecurityEvent,
    Custom(String),
}

/// Audit result
#[derive(Debug, Clone, PartialEq)]
pub enum AuditResult {
    Success,
    Failure,
    Unauthorized,
    Forbidden,
    Error,
}

/// Log storage trait
pub trait LogStorage {
    fn store(&self, event: &AuditEvent) -> DeviceResult<()>;
    fn query(&self, criteria: &QueryCriteria) -> DeviceResult<Vec<AuditEvent>>;
    fn purge_old_logs(&self, before: SystemTime) -> DeviceResult<usize>;
}

/// Query criteria for log retrieval
#[derive(Debug, Clone)]
pub struct QueryCriteria {
    pub start_time: Option<SystemTime>,
    pub end_time: Option<SystemTime>,
    pub user_id: Option<String>,
    pub event_type: Option<AuditEventType>,
    pub resource: Option<String>,
    pub result: Option<AuditResult>,
    pub limit: Option<usize>,
}

impl SessionManager {
    pub fn new(config: SessionConfig, auth_provider: Box<dyn AuthProvider + Send + Sync>) -> Self {
        Self {
            active_sessions: HashMap::new(),
            session_config: config,
            auth_provider,
            permission_manager: PermissionManager::new(),
        }
    }

    pub async fn create_session(&mut self, credentials: &Credentials) -> DeviceResult<UserSession> {
        // Authenticate user
        let auth_result = self.auth_provider.authenticate(credentials)?;

        if !auth_result.success {
            return Err(crate::DeviceError::APIError(
                "Authentication failed".to_string(),
            ));
        }

        let user_info = auth_result
            .user_info
            .ok_or_else(|| crate::DeviceError::APIError("Missing user information".to_string()))?;

        // Create session
        let session_id = self.generate_session_id();
        let expires_at = SystemTime::now() + self.session_config.session_timeout;

        let permissions = self.permission_manager.get_user_permissions(&user_info)?;
        let preferences = self.load_user_preferences(&user_info.user_id).await?;

        let session = UserSession {
            session_id: session_id.clone(),
            user_id: user_info.user_id.clone(),
            permissions,
            preferences,
            last_activity: SystemTime::now(),
            session_data: HashMap::new(),
            auth_token: auth_result.access_token,
            expires_at,
        };

        // Check concurrent session limit
        self.enforce_session_limits(&user_info.user_id)?;

        self.active_sessions
            .insert(session_id.clone(), session.clone());

        Ok(session)
    }

    pub fn validate_session(&mut self, session_id: &str) -> DeviceResult<&mut UserSession> {
        // First check if session exists and is not expired
        if let Some(session) = self.active_sessions.get(session_id) {
            if SystemTime::now() > session.expires_at {
                self.active_sessions.remove(session_id);
                return Err(crate::DeviceError::APIError("Session expired".to_string()));
            }
        } else {
            return Err(crate::DeviceError::APIError("Invalid session".to_string()));
        }

        // Now get mutable reference and update last activity
        let session = self.active_sessions.get_mut(session_id).unwrap();
        session.last_activity = SystemTime::now();

        Ok(session)
    }

    pub fn terminate_session(&mut self, session_id: &str) -> DeviceResult<()> {
        if let Some(session) = self.active_sessions.remove(session_id) {
            // Logout from auth provider if token exists
            if let Some(token) = &session.auth_token {
                let _ = self.auth_provider.logout(token);
            }
        }

        Ok(())
    }

    pub fn cleanup_expired_sessions(&mut self) -> DeviceResult<usize> {
        let now = SystemTime::now();
        let expired_sessions: Vec<String> = self
            .active_sessions
            .iter()
            .filter(|(_, session)| now > session.expires_at)
            .map(|(id, _)| id.clone())
            .collect();

        let count = expired_sessions.len();
        for session_id in expired_sessions {
            self.active_sessions.remove(&session_id);
        }

        Ok(count)
    }

    pub async fn update_user_preferences(
        &mut self,
        user_id: &str,
        preferences: UserPreferences,
    ) -> DeviceResult<()> {
        // Update preferences for all active sessions of this user
        for session in self.active_sessions.values_mut() {
            if session.user_id == user_id {
                session.preferences = preferences.clone();
            }
        }

        // Persist preferences
        self.save_user_preferences(user_id, &preferences).await?;

        Ok(())
    }

    pub fn get_session_statistics(&self) -> SessionStatistics {
        let total_sessions = self.active_sessions.len();
        let mut sessions_by_user = HashMap::new();
        let mut recent_activity_count = 0;
        let recent_threshold = SystemTime::now() - Duration::from_secs(5 * 60);

        for session in self.active_sessions.values() {
            *sessions_by_user.entry(session.user_id.clone()).or_insert(0) += 1;

            if session.last_activity > recent_threshold {
                recent_activity_count += 1;
            }
        }

        SessionStatistics {
            total_active_sessions: total_sessions,
            sessions_by_user,
            recent_activity_count,
            average_session_duration: self.calculate_average_session_duration(),
        }
    }

    fn generate_session_id(&self) -> String {
        // Generate secure session ID
        format!(
            "session_{}",
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        )
    }

    fn enforce_session_limits(&mut self, user_id: &str) -> DeviceResult<()> {
        let user_session_count = self
            .active_sessions
            .values()
            .filter(|s| s.user_id == user_id)
            .count();

        if user_session_count >= self.session_config.max_concurrent_sessions {
            // Remove oldest session for this user
            if let Some((oldest_session_id, _)) = self
                .active_sessions
                .iter()
                .filter(|(_, s)| s.user_id == user_id)
                .min_by_key(|(_, s)| s.last_activity)
                .map(|(id, s)| (id.clone(), s.clone()))
            {
                self.active_sessions.remove(&oldest_session_id);
            }
        }

        Ok(())
    }

    async fn load_user_preferences(&self, user_id: &str) -> DeviceResult<UserPreferences> {
        // Simplified preference loading - in real implementation, load from database
        Ok(UserPreferences::default())
    }

    async fn save_user_preferences(
        &self,
        user_id: &str,
        preferences: &UserPreferences,
    ) -> DeviceResult<()> {
        // Simplified preference saving - in real implementation, save to database
        Ok(())
    }

    fn calculate_average_session_duration(&self) -> Duration {
        if self.active_sessions.is_empty() {
            return Duration::from_secs(0);
        }

        let now = SystemTime::now();
        let total_duration: Duration = self
            .active_sessions
            .values()
            .map(|s| {
                now.duration_since(s.last_activity)
                    .unwrap_or(Duration::from_secs(0))
            })
            .sum();

        total_duration / self.active_sessions.len() as u32
    }
}

/// Session statistics
#[derive(Debug, Clone)]
pub struct SessionStatistics {
    pub total_active_sessions: usize,
    pub sessions_by_user: HashMap<String, usize>,
    pub recent_activity_count: usize,
    pub average_session_duration: Duration,
}

impl PermissionManager {
    pub fn new() -> Self {
        Self {
            role_definitions: Self::create_default_roles(),
            permission_cache: HashMap::new(),
            access_policies: Vec::new(),
        }
    }

    pub fn get_user_permissions(&mut self, user_info: &UserInfo) -> DeviceResult<Vec<Permission>> {
        // Check cache first
        if let Some(cached_permissions) = self.permission_cache.get(&user_info.user_id) {
            return Ok(cached_permissions.clone());
        }

        // Calculate permissions from roles
        let mut permissions = Vec::new();
        for role in &user_info.roles {
            if let Some(role_def) = self.role_definitions.get(&role.role_id) {
                permissions.extend(role_def.permissions.clone());
            }
        }

        // Remove duplicates
        permissions.sort();
        permissions.dedup();

        // Cache permissions
        self.permission_cache
            .insert(user_info.user_id.clone(), permissions.clone());

        Ok(permissions)
    }

    pub fn check_permission(
        &self,
        user_permissions: &[Permission],
        required_permission: &Permission,
    ) -> bool {
        user_permissions.contains(required_permission)
            || user_permissions.contains(&Permission::SystemAdmin)
    }

    pub fn evaluate_access_policy(
        &self,
        user_info: &UserInfo,
        resource: &str,
        action: &str,
    ) -> DeviceResult<bool> {
        for policy in &self.access_policies {
            if policy.resource == resource && policy.action == action {
                if self.evaluate_conditions(&policy.conditions, user_info)? {
                    return Ok(policy.effect == PolicyEffect::Allow);
                }
            }
        }

        // Default deny
        Ok(false)
    }

    fn create_default_roles() -> HashMap<String, Role> {
        let mut roles = HashMap::new();

        roles.insert(
            "admin".to_string(),
            Role {
                role_id: "admin".to_string(),
                role_name: "Administrator".to_string(),
                permissions: vec![Permission::SystemAdmin],
                description: "Full system access".to_string(),
            },
        );

        roles.insert(
            "viewer".to_string(),
            Role {
                role_id: "viewer".to_string(),
                role_name: "Viewer".to_string(),
                permissions: vec![Permission::ReadDashboard, Permission::ViewReports],
                description: "Read-only access".to_string(),
            },
        );

        roles.insert(
            "operator".to_string(),
            Role {
                role_id: "operator".to_string(),
                role_name: "Operator".to_string(),
                permissions: vec![
                    Permission::ReadDashboard,
                    Permission::WriteDashboard,
                    Permission::ManageAlerts,
                    Permission::ExportData,
                    Permission::ViewReports,
                ],
                description: "Operational access".to_string(),
            },
        );

        roles
    }

    fn evaluate_conditions(
        &self,
        conditions: &[AccessCondition],
        user_info: &UserInfo,
    ) -> DeviceResult<bool> {
        for condition in conditions {
            if !self.evaluate_single_condition(condition, user_info)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn evaluate_single_condition(
        &self,
        condition: &AccessCondition,
        user_info: &UserInfo,
    ) -> DeviceResult<bool> {
        match &condition.condition_type {
            ConditionType::UserRole => {
                let has_role = user_info.roles.iter().any(|r| r.role_id == condition.value);
                Ok(match condition.operator {
                    ConditionOperator::Equals => has_role,
                    ConditionOperator::NotEquals => !has_role,
                    _ => false,
                })
            }
            ConditionType::UserGroup => {
                let in_group = user_info.groups.contains(&condition.value);
                Ok(match condition.operator {
                    ConditionOperator::Equals => in_group,
                    ConditionOperator::NotEquals => !in_group,
                    _ => false,
                })
            }
            _ => Ok(true), // Simplified - other conditions not implemented
        }
    }
}

impl AuditLogger {
    pub fn new(config: AuditLogConfig, storage: Box<dyn LogStorage + Send + Sync>) -> Self {
        Self {
            log_config: config,
            log_storage: storage,
            log_buffer: Vec::new(),
        }
    }

    pub fn log_event(&mut self, event: AuditEvent) -> DeviceResult<()> {
        if !self.log_config.enabled {
            return Ok(());
        }

        // Add to buffer
        self.log_buffer.push(event);

        // Flush if buffer is full
        if self.log_buffer.len() >= 100 {
            self.flush_buffer()?;
        }

        Ok(())
    }

    pub fn flush_buffer(&mut self) -> DeviceResult<()> {
        for event in &self.log_buffer {
            self.log_storage.store(event)?;
        }
        self.log_buffer.clear();
        Ok(())
    }

    pub fn query_logs(&self, criteria: &QueryCriteria) -> DeviceResult<Vec<AuditEvent>> {
        self.log_storage.query(criteria)
    }
}

// Default implementations
impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            dashboard_layout: "grid".to_string(),
            default_time_range: "last_hour".to_string(),
            chart_preferences: ChartPreferences::default(),
            notification_preferences: NotificationPreferences::default(),
            display_preferences: DisplayPreferences::default(),
            custom_preferences: HashMap::new(),
        }
    }
}

impl Default for ChartPreferences {
    fn default() -> Self {
        Self {
            preferred_chart_types: vec!["line".to_string(), "bar".to_string()],
            color_scheme: "scientific".to_string(),
            animation_enabled: true,
            interactive_features: true,
            default_aggregation: "minute".to_string(),
            refresh_interval: 30,
        }
    }
}

impl Default for NotificationPreferences {
    fn default() -> Self {
        Self {
            email_notifications: true,
            browser_notifications: false,
            slack_notifications: false,
            notification_frequency: NotificationFrequency::Immediate,
            alert_thresholds: HashMap::new(),
            quiet_hours: None,
        }
    }
}

impl Default for DisplayPreferences {
    fn default() -> Self {
        Self {
            theme: "light".to_string(),
            font_size: "medium".to_string(),
            density: DisplayDensity::Normal,
            sidebar_collapsed: false,
            show_tooltips: true,
            language: "en".to_string(),
            timezone: "UTC".to_string(),
        }
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            session_timeout: Duration::from_secs(8 * 3600),
            max_concurrent_sessions: 5,
            require_authentication: true,
            enable_session_persistence: false,
            cookie_settings: CookieSettings {
                secure: true,
                http_only: true,
                same_site: SameSitePolicy::Strict,
                domain: None,
                path: "/".to_string(),
                max_age: Duration::from_secs(8 * 3600),
            },
            security_settings: SecuritySettings {
                csrf_protection: true,
                rate_limiting: true,
                ip_whitelist: None,
                require_https: true,
                enable_audit_logging: true,
            },
        }
    }
}

impl Default for AuditLogConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_level: AuditLogLevel::Info,
            retention_period: Duration::from_secs(90 * 24 * 3600),
            include_sensitive_data: false,
            encryption_enabled: true,
        }
    }
}
