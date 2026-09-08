## Authentication-related edge cases

Authentication and authorization are related but are not interchangeable.

A request can be authenticated successfully while still being rejected because
the API key does not have sufficient scope. In that case, the key itself may be
valid even though the requested operation is not permitted.

For example:

- A valid `read` key can authenticate successfully but cannot perform a
  write operation.
- A valid `write` key can authenticate successfully but cannot perform an
  administrative operation.
- An invalid, missing, or revoked key is an authentication failure and returns
  `401 Unauthorized`.

Do not infer the required scope from the HTTP status code alone. The scope
requirement depends on the operation being performed.

### Similar-looking status codes

The following cases are intentionally easy to confuse:

| Situation | HTTP status | Meaning |
|---|---:|---|
| API key missing | `401` | Authentication failure |
| API key invalid | `401` | Authentication failure |
| API key revoked | `401` | Authentication failure |
| API key valid but insufficient scope | `403` | Authorization failure |
| Rate limit exceeded | `429` | Too many requests |

A `429` response does not indicate that the API key is invalid. It indicates
that the request rate has exceeded the permitted limit.

When a `429` is returned, clients should read the `Retry-After` response
header and wait before retrying. Retrying immediately with the same request
does not resolve the rate-limit condition.

### Access key terminology

The terms `API key`, `access key`, and `bearer token` may appear in application
configuration as equivalent names for the credential used to authenticate an
API request.

However, these terms should not be interpreted as three different credentials.
The AcmeSDK API expects the credential to be supplied using the documented
Authorization header format.

### Scope and key lifecycle

Scope is a property of the API key, not of an individual request.

A key created with `read` scope does not become a `write` key because the
request contains a different endpoint or HTTP method.

Similarly, rotating a key does not change the scope of an existing key.
Rotation creates a replacement key, which may be configured with the required
scope.

API keys do not expire automatically. Rotation is recommended every 90 days,
but the recommendation should not be interpreted as an automatic expiration
rule.

During zero-downtime rotation, both the old and replacement keys may be active
at the same time. The old key remains usable until it is explicitly revoked.

Once revoked, a key cannot be used again.

### Common configuration mistakes

The following configurations should not be confused:

```text
Authorization: Bearer YOUR_API_KEY