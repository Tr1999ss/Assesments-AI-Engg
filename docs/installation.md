# Installation & Quickstart

## Requirements
AcmeSDK requires Python 3.9 or later. It is tested on Python 3.9 through 3.12.

## Installing the SDK
Install via pip:

```
pip install acme-sdk
```

For async support, install the extras:

```
pip install acme-sdk[async]
```

## Quickstart
```python
from acme_sdk import Client

client = Client(api_key="YOUR_API_KEY")
result = client.jobs.create(name="my-first-job")
print(result.id)
```

## Configuring the base URL
By default the SDK points to `https://api.acme.dev`. To use a different region or a self-hosted
instance, pass `base_url` when constructing the client:

```python
client = Client(api_key="YOUR_API_KEY", base_url="https://eu.api.acme.dev")
```

## Verifying your installation
Run `acme-sdk doctor` from the command line to check that your environment is configured correctly.
It verifies your Python version, checks that your API key is valid, and confirms network connectivity
to the API.
