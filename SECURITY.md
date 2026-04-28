# Security Policy

## Supported Versions

The following versions of QuantRS2 receive security patches:

| Version | Supported |
|---------|-----------|
| 0.2.x   | Yes       |
| 0.1.x   | Yes (LTS) |
| < 0.1   | No        |

---

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

If you discover a security vulnerability in QuantRS2, please report it
privately by sending an email to:

**kitahata@gmail.com**

Use the subject line:

```
[SECURITY] QuantRS2 vulnerability
```

Include in your report:
- A description of the vulnerability and its potential impact
- Steps to reproduce or a proof-of-concept (if safe to share)
- The version(s) of QuantRS2 affected
- Any suggested mitigations or fixes, if you have them

---

## Disclosure Timeline

Once a report is received, the project team will follow this timeline:

| Milestone | Target |
|-----------|--------|
| Acknowledgment of receipt | Within 72 hours |
| Initial assessment and severity triage | Within 7 days |
| Patch developed and reviewed | Within 90 days |
| Coordinated public disclosure | After patch is available |

If a patch will take longer than 90 days due to complexity, the reporter
will be notified of the delay and kept informed of progress.

---

## Embargo Policy

Reporters agree not to disclose the vulnerability publicly until the
maintainers have released a patch or have explicitly communicated that
disclosure may proceed. This coordinated disclosure approach protects users
who have not yet updated their installations.

The maintainers will credit the reporter by name (or handle) in the security
advisory unless the reporter requests anonymity. If anonymity is requested,
the advisory will note that the reporter chose to remain anonymous.

---

## Scope

### In Scope

The following components are in scope for security reports:

- Quantum circuit core (`quantrs2-core`)
- Sampler and optimization algorithms (`quantrs2-anneal`, `quantrs2-tytan`)
- Python bindings (`quantrs2-py`)
- The public API surface of all published `quantrs2-*` crates
- Dependency vulnerabilities that directly affect the default build of
  any published crate

### Out of Scope

The following are generally out of scope:

- Vulnerabilities in third-party dependencies that are not yet public
  knowledge (report these upstream to the dependency maintainer)
- Issues in demo scripts, examples, or documentation that cannot be
  exploited to affect user data or systems
- Documentation typos or inaccuracies that do not have security implications
- Denial-of-service conditions caused by pathologically large quantum circuits
  (QuantRS2 does not guarantee resource limits for adversarial inputs)
- Issues that require physical access to the machine running QuantRS2
- Social engineering attacks against project maintainers

---

## Contact

For all security-related matters, contact:

**kitahata@gmail.com**
