# Authentication

## Overview
All requests to the AcmeSDK API must be authenticated using an API key. You can generate a key from
the Developer Dashboard under Settings > API Keys.

## Using your API key
Pass your API key in the `Authorization` header of every request:

```
Authorization: Bearer YOUR_API_KEY
```

Requests without a valid key will receive a `401 Unauthorized` response.

## Key rotation
API keys do not expire automatically, but we recommend rotating them every 90 days. You can have up to
two active keys at a time to allow zero-downtime rotation: generate a new key, update your application,
then revoke the old key.

## Rate limits
Authenticated requests are limited to 1000 requests per minute per API key. If you exceed this limit,
the API returns a `429 Too Many Requests` response with a `Retry-After` header indicating how many
seconds to wait before retrying.

## Scopes
API keys can be scoped to `read`, `write`, or `admin`. Scope is set when the key is created and cannot
be changed afterward — generate a new key if you need different permissions.
