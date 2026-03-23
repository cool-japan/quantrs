//! # ItsmConfig - Trait Implementations
//!
//! This module contains trait implementations for `ItsmConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{BTreeMap, HashMap, VecDeque};

use super::types::{
    AuthenticationMethod, IncidentPriority, IncidentTemplate, ItsmConfig, ItsmPlatform,
};

impl Default for ItsmConfig {
    fn default() -> Self {
        let mut incident_templates = HashMap::new();
        incident_templates.insert(
            "slo_breach".to_string(),
            IncidentTemplate {
                priority: IncidentPriority::High,
                category: "Performance".to_string(),
                assignment_group: "QuantumOps".to_string(),
                escalation_path: vec!["L2".to_string(), "L3".to_string()],
            },
        );
        Self {
            platform: ItsmPlatform::ServiceNow,
            endpoint: "https://company.service-now.com/api".to_string(),
            authentication: AuthenticationMethod::OAuth2(String::new()),
            incident_templates,
        }
    }
}
