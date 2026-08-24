# Jobs API

## Creating a job
```python
client.jobs.create(name="my-job", payload={"key": "value"}, priority="normal")
```

`priority` accepts `low`, `normal`, or `high`. Jobs default to `normal` priority if not specified.

## Job statuses
A job moves through the following statuses: `queued` -> `running` -> `completed` or `failed`.
You can poll `client.jobs.get(job_id)` to check the current status, or subscribe to webhook
events for status changes.

## Retrying failed jobs
Failed jobs are not retried automatically. Call `client.jobs.retry(job_id)` to requeue a failed job.
A job can be retried up to 5 times; after that it is permanently marked `failed` and must be
recreated.

## Deleting jobs
Jobs can only be deleted once they reach a terminal state (`completed` or `failed`).
`client.jobs.delete(job_id)` will raise a `JobStillRunningError` if called on an active job.

## Webhooks
Register a webhook URL in the dashboard to receive `job.completed` and `job.failed` events.
Webhook payloads are signed with your webhook secret; verify the `X-Acme-Signature` header before
trusting the payload.
