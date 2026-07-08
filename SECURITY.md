# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

Please report security vulnerabilities by opening a private security advisory on
GitHub (Security → Advisories → New draft advisory) rather than a public issue.

You can expect an initial response within 7 days. Please include:

- A description of the vulnerability and its impact.
- Steps to reproduce.
- Affected version(s).

## Security posture

- The API container runs as a non-root user.
- Rate limiting (200 req/min per IP) mitigates brute-force and scraping abuse.
- All request payloads are validated with Pydantic before reaching model code.
- No secrets are stored in the repository; configuration comes from environment
  variables (see `.env.example`).
