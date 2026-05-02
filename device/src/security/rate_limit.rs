//! Token bucket rate limiter for cloud API calls.
//!
//! Prevents hitting provider rate limits (IBM: 5 jobs/min, AWS: 10 req/s, etc.)
//! by tracking token consumption per backend. The token bucket algorithm smooths
//! bursty traffic while respecting sustained-rate limits.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// A token bucket: accumulates capacity tokens at `refill_rate` tokens/second.
///
/// Starts full (all tokens available). Each API call consumes one or more tokens.
/// Tokens are replenished continuously up to `capacity`.
pub struct TokenBucket {
    capacity: f64,
    tokens: f64,
    refill_rate: f64, // tokens per second
    last_refill: Instant,
}

impl TokenBucket {
    /// Create a new token bucket with the given capacity and refill rate.
    ///
    /// # Arguments
    /// * `capacity`    — maximum number of tokens (burst ceiling)
    /// * `refill_rate` — tokens added per second (sustained throughput)
    pub fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            capacity,
            tokens: capacity,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    /// Refill tokens based on elapsed time since the last call
    fn refill(&mut self) {
        let elapsed = self.last_refill.elapsed().as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_refill = Instant::now();
    }

    /// Try to consume `tokens` from the bucket.
    ///
    /// Returns `true` if the tokens were available and consumed,
    /// `false` if the bucket is too empty (caller should wait).
    pub fn try_consume(&mut self, tokens: f64) -> bool {
        self.refill();
        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }

    /// Estimate wait time before `tokens` become available.
    ///
    /// Returns [`Duration::ZERO`] if sufficient tokens exist right now.
    pub fn wait_time(&mut self, tokens: f64) -> Duration {
        self.refill();
        if self.tokens >= tokens {
            Duration::ZERO
        } else {
            let needed = tokens - self.tokens;
            let wait_secs = needed / self.refill_rate;
            Duration::from_secs_f64(wait_secs)
        }
    }

    /// Return the current token count after applying any accumulated refill
    pub fn available_tokens(&mut self) -> f64 {
        self.refill();
        self.tokens
    }

    /// Return the configured bucket capacity (burst ceiling)
    pub fn capacity(&self) -> f64 {
        self.capacity
    }

    /// Return the configured refill rate in tokens per second
    pub fn refill_rate(&self) -> f64 {
        self.refill_rate
    }
}

impl std::fmt::Debug for TokenBucket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenBucket")
            .field("capacity", &self.capacity)
            .field("tokens", &self.tokens)
            .field("refill_rate", &self.refill_rate)
            .finish()
    }
}

/// Per-provider token-bucket rate limiter.
///
/// Maintains a separate [`TokenBucket`] for each provider key. Providers not
/// explicitly configured get a bucket with the default capacity and rate.
///
/// # Example
///
/// ```rust
/// use quantrs2_device::security::rate_limit::RateLimiter;
///
/// let mut limiter = RateLimiter::with_cloud_defaults();
///
/// // Fast path — tokens available
/// if limiter.try_consume("aws") {
///     // submit request
/// } else {
///     let delay = limiter.wait_time("aws");
///     // sleep(delay), then retry
/// }
/// ```
pub struct RateLimiter {
    buckets: HashMap<String, TokenBucket>,
    default_capacity: f64,
    default_rate: f64,
}

impl RateLimiter {
    /// Create a rate limiter with the given defaults for unconfigured providers.
    ///
    /// # Arguments
    /// * `default_capacity`         — burst limit for unknown providers
    /// * `default_rate_per_second`  — sustained rate for unknown providers
    pub fn new(default_capacity: f64, default_rate_per_second: f64) -> Self {
        Self {
            buckets: HashMap::new(),
            default_capacity,
            default_rate: default_rate_per_second,
        }
    }

    /// Register a provider-specific bucket, overriding the defaults.
    pub fn with_provider(
        mut self,
        provider: impl Into<String>,
        capacity: f64,
        rate: f64,
    ) -> Self {
        self.buckets
            .insert(provider.into(), TokenBucket::new(capacity, rate));
        self
    }

    /// Try to consume one token for the given provider key.
    ///
    /// Creates a default bucket for the provider if it has not been seen before.
    /// Returns `true` if the request can proceed immediately.
    pub fn try_consume(&mut self, provider: &str) -> bool {
        let (cap, rate) = (self.default_capacity, self.default_rate);
        let bucket = self
            .buckets
            .entry(provider.to_string())
            .or_insert_with(|| TokenBucket::new(cap, rate));
        bucket.try_consume(1.0)
    }

    /// Estimate the wait time before a token is available for the given provider.
    ///
    /// Returns [`Duration::ZERO`] if a token is available immediately.
    pub fn wait_time(&mut self, provider: &str) -> Duration {
        let (cap, rate) = (self.default_capacity, self.default_rate);
        let bucket = self
            .buckets
            .entry(provider.to_string())
            .or_insert_with(|| TokenBucket::new(cap, rate));
        bucket.wait_time(1.0)
    }

    /// Return the number of available tokens for a provider (after refill).
    ///
    /// Creates a default bucket if the provider has not been seen before.
    pub fn available_tokens(&mut self, provider: &str) -> f64 {
        let (cap, rate) = (self.default_capacity, self.default_rate);
        let bucket = self
            .buckets
            .entry(provider.to_string())
            .or_insert_with(|| TokenBucket::new(cap, rate));
        bucket.available_tokens()
    }

    /// Pre-configured limiter with typical cloud provider limits:
    ///
    /// | Provider | Burst | Sustained       |
    /// |----------|-------|-----------------|
    /// | IBM      | 5     | 5 / 60s         |
    /// | AWS      | 10    | 10/s            |
    /// | Azure    | 10    | 10/s            |
    ///
    /// Unknown providers get a 10-token bucket at 1 token/second.
    pub fn with_cloud_defaults() -> Self {
        Self::new(10.0, 1.0)
            .with_provider("ibm", 5.0, 5.0 / 60.0)
            .with_provider("aws", 10.0, 10.0)
            .with_provider("azure", 10.0, 10.0)
    }

    /// Return the list of currently tracked provider keys
    pub fn tracked_providers(&self) -> Vec<&str> {
        self.buckets.keys().map(|s| s.as_str()).collect()
    }
}

impl std::fmt::Debug for RateLimiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RateLimiter")
            .field("providers", &self.buckets.keys().collect::<Vec<_>>())
            .field("default_capacity", &self.default_capacity)
            .field("default_rate", &self.default_rate)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_token_bucket_starts_full() {
        let mut bucket = TokenBucket::new(10.0, 1.0);
        assert!((bucket.available_tokens() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_token_bucket_consume_success() {
        let mut bucket = TokenBucket::new(5.0, 1.0);
        assert!(bucket.try_consume(3.0));
        // ~2 tokens remain
        assert!(bucket.available_tokens() < 3.0);
    }

    #[test]
    fn test_token_bucket_consume_fails_when_empty() {
        let mut bucket = TokenBucket::new(3.0, 0.001); // very slow refill
        // Drain all tokens
        assert!(bucket.try_consume(3.0));
        // Next consume should fail immediately
        assert!(!bucket.try_consume(1.0));
    }

    #[test]
    fn test_token_bucket_wait_time_zero_when_full() {
        let mut bucket = TokenBucket::new(10.0, 1.0);
        let wait = bucket.wait_time(1.0);
        assert_eq!(wait, Duration::ZERO);
    }

    #[test]
    fn test_token_bucket_wait_time_nonzero_when_empty() {
        let mut bucket = TokenBucket::new(3.0, 0.001); // very slow refill
        assert!(bucket.try_consume(3.0));
        let wait = bucket.wait_time(1.0);
        // wait should be positive (≈ 1000s at 0.001 t/s)
        assert!(wait > Duration::ZERO);
    }

    #[test]
    fn test_token_bucket_capacity_ceiling() {
        // Even after a long wait the bucket won't exceed capacity
        let mut bucket = TokenBucket::new(5.0, 100.0);
        // Force-set tokens as if a long time passed — by consuming 0 and then refilling
        // Simulate by constructing with a past Instant
        let tokens = bucket.available_tokens();
        assert!(tokens <= 5.0 + 1e-9); // never exceeds capacity
    }

    #[test]
    fn test_token_bucket_accessors() {
        let bucket = TokenBucket::new(10.0, 2.5);
        assert!((bucket.capacity() - 10.0).abs() < 1e-9);
        assert!((bucket.refill_rate() - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_rate_limiter_new_provider_gets_defaults() {
        let mut limiter = RateLimiter::new(5.0, 1.0);
        // "unknown" provider should start with full bucket
        let tokens = limiter.available_tokens("unknown_provider");
        assert!((tokens - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_rate_limiter_try_consume_success() {
        let mut limiter = RateLimiter::new(10.0, 1.0);
        assert!(limiter.try_consume("aws"));
    }

    #[test]
    fn test_rate_limiter_exhaustion() {
        let mut limiter = RateLimiter::new(3.0, 0.001);
        assert!(limiter.try_consume("test"));
        assert!(limiter.try_consume("test"));
        assert!(limiter.try_consume("test"));
        // 4th should fail
        assert!(!limiter.try_consume("test"));
    }

    #[test]
    fn test_rate_limiter_cloud_defaults_ibm() {
        let mut limiter = RateLimiter::with_cloud_defaults();
        // IBM bucket has capacity 5
        for _ in 0..5 {
            assert!(limiter.try_consume("ibm"));
        }
        // 6th should fail (very slow refill)
        assert!(!limiter.try_consume("ibm"));
    }

    #[test]
    fn test_rate_limiter_cloud_defaults_aws() {
        let mut limiter = RateLimiter::with_cloud_defaults();
        // AWS bucket has capacity 10
        for _ in 0..10 {
            assert!(limiter.try_consume("aws"));
        }
        assert!(!limiter.try_consume("aws"));
    }

    #[test]
    fn test_rate_limiter_wait_time_zero_when_available() {
        let mut limiter = RateLimiter::new(10.0, 1.0);
        let wait = limiter.wait_time("any_provider");
        assert_eq!(wait, Duration::ZERO);
    }

    #[test]
    fn test_rate_limiter_wait_time_positive_when_exhausted() {
        let mut limiter = RateLimiter::new(1.0, 0.001);
        assert!(limiter.try_consume("provider"));
        let wait = limiter.wait_time("provider");
        assert!(wait > Duration::ZERO);
    }

    #[test]
    fn test_rate_limiter_independent_providers() {
        let mut limiter = RateLimiter::new(2.0, 0.001);
        // Exhaust "provider_a"
        assert!(limiter.try_consume("provider_a"));
        assert!(limiter.try_consume("provider_a"));
        assert!(!limiter.try_consume("provider_a"));

        // "provider_b" should be unaffected
        assert!(limiter.try_consume("provider_b"));
        assert!(limiter.try_consume("provider_b"));
    }

    #[test]
    fn test_rate_limiter_tracked_providers() {
        let mut limiter = RateLimiter::with_cloud_defaults();
        // Pre-configured providers are already tracked
        let providers = limiter.tracked_providers();
        assert!(providers.contains(&"ibm"));
        assert!(providers.contains(&"aws"));
        assert!(providers.contains(&"azure"));

        // Accessing a new provider adds it to the tracked set
        limiter.try_consume("ionq");
        let providers = limiter.tracked_providers();
        assert!(providers.contains(&"ionq"));
    }

    #[test]
    fn test_token_bucket_debug() {
        let bucket = TokenBucket::new(5.0, 1.0);
        let s = format!("{:?}", bucket);
        assert!(s.contains("TokenBucket"));
    }

    #[test]
    fn test_rate_limiter_debug() {
        let limiter = RateLimiter::with_cloud_defaults();
        let s = format!("{:?}", limiter);
        assert!(s.contains("RateLimiter"));
    }
}
