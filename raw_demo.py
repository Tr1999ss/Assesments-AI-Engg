"""
raw_demo.py — talk to the MCP server with raw JSON, no MCP library, so you see
exactly what travels over the wire. Run it once and MCP stops being a mystery.

    python raw_demo.py

Sequence (this IS the protocol):
    1. initialize            client says hello + protocol version, server replies with capabilities
    2. notifications/initialized   client confirms (a notification: no id, no reply)
    3. tools/list            "what can you do?"  -> server returns tool names + JSON schemas
    4. tools/call            "do it"             -> server returns the result
"""

import json
import subprocess
import sys

proc = subprocess.Popen(
    [sys.executable, "mcp_server.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,  # server logs/banner go to stderr; hide them here
    text=True,
    encoding="utf-8",
)


def send(msg):
    print("CLIENT ->", json.dumps(msg)[:300])
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()


def receive():
    line = proc.stdout.readline()
    print("SERVER <-", line.strip()[:500])
    print()
    return json.loads(line)


send({
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "raw-demo", "version": "0.1"},
    },
})
receive()

send({"jsonrpc": "2.0", "method": "notifications/initialized"})

send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
receive()

send({
    "jsonrpc": "2.0", "id": 3, "method": "tools/call",
    "params": {"name": "search_docs", "arguments": {"question": "What does a 429 mean?", "top_k": 2}},
})
receive()

proc.terminate()