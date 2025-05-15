use chrono::{DateTime, Utc};
use hmac::{Hmac, Mac};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

type HmacSha256 = Hmac<Sha256>;

/// AWS request region information
#[derive(Debug)]
pub struct AwsRegion {
    /// Region name (e.g., "us-east-1")
    pub name: String,
    /// AWS service name (e.g., "braket")
    pub service: String,
}

/// Generate a signature for AWS Signature Version 4
pub struct AwsSignatureV4;

impl AwsSignatureV4 {
    /// Sign a request with AWS Signature Version 4
    pub fn sign_request(
        method: &str,
        path: &str,
        query: &str,
        headers: &mut HeaderMap,
        body: &[u8],
        access_key: &str,
        secret_key: &str,
        region: &AwsRegion,
        time: &DateTime<Utc>,
    ) {
        // Step 1: Create canonical request
        let canonical_request = Self::create_canonical_request(method, path, query, headers, body);

        // Step 2: Create string to sign
        let string_to_sign = Self::create_string_to_sign(&canonical_request, &region, time);

        // Step 3: Calculate signature
        let signature = Self::calculate_signature(&string_to_sign, secret_key, &region, time);

        // Step 4: Add signature to headers
        let credential = format!(
            "{}/{}/{}/{}/aws4_request",
            access_key,
            time.format("%Y%m%d"),
            region.name,
            region.service
        );

        let signed_headers = Self::get_signed_headers(headers);

        let auth_header = format!(
            "AWS4-HMAC-SHA256 Credential={},SignedHeaders={},Signature={}",
            credential, signed_headers, signature
        );

        headers.insert(
            HeaderName::from_static("authorization"),
            HeaderValue::from_str(&auth_header).unwrap(),
        );
    }

    /// Create canonical request for signing
    fn create_canonical_request(
        method: &str,
        path: &str,
        query: &str,
        headers: &HeaderMap,
        body: &[u8],
    ) -> String {
        // 1. Start with the request method
        let mut canonical = method.to_uppercase();
        canonical.push('\n');

        // 2. Add the canonical URI
        canonical.push_str(&Self::canonical_uri(path));
        canonical.push('\n');

        // 3. Add the canonical query string
        canonical.push_str(&Self::canonical_query_string(query));
        canonical.push('\n');

        // 4. Add the canonical headers
        let (canonical_headers, signed_headers) = Self::canonical_headers(headers);
        canonical.push_str(&canonical_headers);
        canonical.push('\n');

        // 5. Add the signed headers
        canonical.push_str(&signed_headers);
        canonical.push('\n');

        // 6. Add the hashed payload
        let payload_hash = Self::hash_payload(body);
        canonical.push_str(&payload_hash);

        canonical
    }

    /// Create canonical URI for signing
    fn canonical_uri(path: &str) -> String {
        let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

        if segments.is_empty() {
            return "/".to_string();
        }

        format!("/{}", segments.join("/"))
    }

    /// Create canonical query string for signing
    fn canonical_query_string(query: &str) -> String {
        if query.is_empty() {
            return String::new();
        }

        let mut params = BTreeMap::new();
        for pair in query.split('&') {
            let mut split = pair.split('=');
            let key = split.next().unwrap_or("");
            let value = split.next().unwrap_or("");

            params.insert(Self::uri_encode(key, true), Self::uri_encode(value, true));
        }

        let mut canonical_query = String::new();
        for (key, value) in params {
            if !canonical_query.is_empty() {
                canonical_query.push('&');
            }
            canonical_query.push_str(&key);
            canonical_query.push('=');
            canonical_query.push_str(&value);
        }

        canonical_query
    }

    /// Create canonical headers for signing
    fn canonical_headers(headers: &HeaderMap) -> (String, String) {
        let mut canonical_headers = String::new();
        let mut signed_headers = String::new();

        let mut header_map = BTreeMap::new();
        for (key, value) in headers.iter() {
            let header_name = key.as_str().to_lowercase();
            let header_value = value.to_str().unwrap_or("").trim();

            if header_name == "host"
                || header_name.starts_with("x-amz-")
                || header_name == "content-type"
            {
                header_map.insert(header_name, header_value.to_owned());
            }
        }

        for (key, value) in &header_map {
            canonical_headers.push_str(key);
            canonical_headers.push(':');
            canonical_headers.push_str(value);
            canonical_headers.push('\n');

            if !signed_headers.is_empty() {
                signed_headers.push(';');
            }
            signed_headers.push_str(key);
        }

        (canonical_headers, signed_headers)
    }

    /// Get signed headers string
    fn get_signed_headers(headers: &HeaderMap) -> String {
        let mut signed_headers = String::new();
        let mut header_names: Vec<String> = headers
            .keys()
            .filter_map(|name| {
                let name = name.as_str().to_lowercase();
                if name == "host" || name.starts_with("x-amz-") || name == "content-type" {
                    Some(name)
                } else {
                    None
                }
            })
            .collect();

        header_names.sort();

        for (i, name) in header_names.iter().enumerate() {
            if i > 0 {
                signed_headers.push(';');
            }
            signed_headers.push_str(name);
        }

        signed_headers
    }

    /// Hash request payload
    fn hash_payload(payload: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(payload);
        let hash = hasher.finalize();

        hex::encode(hash)
    }

    /// Create string to sign
    fn create_string_to_sign(
        canonical_request: &str,
        region: &AwsRegion,
        time: &DateTime<Utc>,
    ) -> String {
        let mut string_to_sign = String::from("AWS4-HMAC-SHA256\n");

        // Add timestamp
        string_to_sign.push_str(&time.format("%Y%m%dT%H%M%SZ").to_string());
        string_to_sign.push('\n');

        // Add credential scope
        string_to_sign.push_str(&format!(
            "{}/{}/{}/aws4_request\n",
            time.format("%Y%m%d"),
            region.name,
            region.service
        ));

        // Add hashed canonical request
        let mut hasher = Sha256::new();
        hasher.update(canonical_request.as_bytes());
        let hash = hasher.finalize();

        string_to_sign.push_str(&hex::encode(hash));

        string_to_sign
    }

    /// Calculate signature
    fn calculate_signature(
        string_to_sign: &str,
        secret_key: &str,
        region: &AwsRegion,
        time: &DateTime<Utc>,
    ) -> String {
        // Create signing key
        let k_secret = format!("AWS4{}", secret_key);
        let k_date = Self::hmac_sha256(
            k_secret.as_bytes(),
            time.format("%Y%m%d").to_string().as_bytes(),
        );
        let k_region = Self::hmac_sha256(&k_date, region.name.as_bytes());
        let k_service = Self::hmac_sha256(&k_region, region.service.as_bytes());
        let k_signing = Self::hmac_sha256(&k_service, b"aws4_request");

        // Calculate the signature
        let signature = Self::hmac_sha256(&k_signing, string_to_sign.as_bytes());

        hex::encode(signature)
    }

    /// Compute HMAC-SHA256
    fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
        let mut mac = HmacSha256::new_from_slice(key).expect("HMAC can take key of any size");
        mac.update(data);
        mac.finalize().into_bytes().to_vec()
    }

    /// URI encode a string
    fn uri_encode(string: &str, encode_slash: bool) -> String {
        let mut result = String::new();

        for byte in string.as_bytes() {
            match byte {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                    result.push(*byte as char);
                }
                b'/' if !encode_slash => {
                    result.push('/');
                }
                _ => {
                    result.push('%');
                    result.push_str(&format!("{:02X}", byte));
                }
            }
        }

        result
    }
}
