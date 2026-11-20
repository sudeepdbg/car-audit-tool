# Security Policy

## Reporting a vulnerability

Please do **not** open a public issue for security problems. Use GitHub's
"Report a vulnerability" (Security tab -> Advisories) on this repository, or
contact the maintainer privately. Include steps to reproduce and the affected
version.

## Deployment guidance

- Run behind a TLS-terminating reverse proxy (nginx, Caddy) before exposing it
  beyond localhost. The app binds to `127.0.0.1` by default.
- Set a strong `JWT_SECRET` and change the generated admin password.
- Rental photos may contain personal data (faces, licence plates). Apply your
  own retention policy; admins can delete audits, which also removes the stored files.
- Login rate limiting is in-memory and per-process; use a proxy-level limiter
  for multi-worker deployments. Failed logins are logged as security events
  and a 429 response carries a `Retry-After` header.
- JWTs are stateless **but revocable**. Every token carries a `ver` claim that
  matches the user's `token_version` column in SQLite. Password changes, role
  changes, user deletion, and `/api/auth/logout` all bump that column, which
  invalidates the user's outstanding access + refresh tokens immediately (no
  8h / 7d grace window). Refresh tokens are not single-use: a rotated token stays valid until it expires or
  the version is bumped, so a stolen one becomes worthless only when the owner
  changes their password or signs out.
- Responses include `Content-Security-Policy`, `X-Frame-Options: DENY`,
  `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, and HSTS.
- For production use a real WSGI server. The recommended single-process option
  (keeps the in-memory login limiter consistent) is waitress, which works on
  Windows too. See README.md for the exact commands.
