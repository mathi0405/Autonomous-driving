# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.2.x | ✅ Active |
| 0.1.x | ⚠️ Security fixes only |
| < 0.1 | ❌ Unsupported |

## Reporting a Vulnerability

Do **not** open a public GitHub issue for security vulnerabilities.

Please report vulnerabilities by emailing the maintainer directly (see the GitHub profile for contact details). Include:

- A concise description of the vulnerability
- Steps to reproduce, if applicable
- Potential impact assessment
- Any suggested mitigations

You will receive an acknowledgement within 72 hours. Confirmed vulnerabilities will be patched and disclosed publicly after a fix is available, following [responsible disclosure](https://en.wikipedia.org/wiki/Coordinated_vulnerability_disclosure) principles.

## Scope

This project is a research codebase, not a production system. The primary security concern is dependency vulnerabilities. We use `pip-audit` in the CI pipeline to detect known CVEs in the dependency tree.
