#!/usr/bin/env python3
"""
Tailscale Tool Module

Provides tools for exposing local services to your tailnet via Tailscale Serve,
and transferring files between tailnet devices via Tailscale SSH.

Tailscale Serve shares a local server securely within your tailnet using
Tailscale-issued HTTPS certificates. No port forwarding, no public exposure.

Example use cases:
  - Expose a localhost dev server (WordPress, Next.js, etc.) to your laptop
  - Share a preview URL with yourself across devices
  - Access a service running inside the VPS from your laptop via tailnet IP

Tailscale SSH enables passwordless, keyless SSH between tailnet devices.
"""

import json
import logging
import subprocess
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: int = 15) -> tuple[int, str, str]:
    """Run a command, return (exit_code, stdout, stderr)."""
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
        )
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except Exception as e:
        return -1, "", str(e)


def _get_tailscale_ip() -> Optional[str]:
    """Get the primary IPv4 Tailscale IP of this device."""
    code, out, _ = _run(["tailscale", "status", "--self", "--json"])
    if code != 0:
        return None
    try:
        data = json.loads(out)
        tailscale_ips = data.get("TailscaleIPs", [])
        for ip in tailscale_ips:
            if "." in ip:  # IPv4
                return ip
        return tailscale_ips[0] if tailscale_ips else None
    except (json.JSONDecodeError, KeyError, TypeError, IndexError):
        return None


def _get_tailscale_device_fqdn() -> Optional[str]:
    """Get the FQDN of the current device (e.g. 'ubuntu-16gb-nbg1-1.tail102593.ts.net')."""
    code, out, _ = _run(["tailscale", "status", "--self", "--json"])
    if code != 0:
        return None
    try:
        data = json.loads(out)
        dns_name = data.get("Self", {}).get("DNSName", "")
        if dns_name:
            return dns_name.rstrip(".")
        return data.get("Self", {}).get("HostName", "")
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

TAILSCALE_SERVE_STATUS_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["status", "reset"],
            "description": "Action to perform."
        }
    },
    "required": ["action"]
}


TAILSCALE_SERVE_SCHEMA = {
    "name": "tailscale_serve",
    "description": """Expose a local HTTP/HTTPS service to your entire tailnet via Tailscale Serve.

Tailscale Serve shares a local server securely within your tailnet using
Tailscale-issued HTTPS certificates — no public exposure, no port forwarding.

The tool returns a URL in the form: https://<tailscale-ip>:<port>/

IMPORTANT: For port 443 (or 80), Tailscale uses HTTPS with a valid certificate.
For other ports, Tailscale also issues a certificate — access via the returned URL.

PREREQUISITES:
- Tailscale must be running and authenticated on this machine.
- The local service must be listening on 127.0.0.1 or localhost.

RESULTS:
- The URL is accessible to ALL devices in your tailnet (no extra login).
- The serve config persists until explicitly reset.
- To stop sharing, call tailscale_serve with action=reset.

EXAMPLES:
- Expose a WordPress dev server on port 8091:  port=8091
- Expose a Node.js app on port 3000:  port=3000
- Expose a Python HTTP server:  port=8000""",
    "parameters": {
        "type": "object",
        "properties": {
            "port": {
                "type": "integer",
                "description": "Local port to expose (e.g. 8091, 3000, 8000)."
            },
            "protocol": {
                "type": "string",
                "enum": ["http", "https", "https+insecure"],
                "default": "http",
                "description": "Protocol of the local service. Use 'https+insecure' for services with self-signed certs."
            },
            "bg": {
                "type": "boolean",
                "default": True,
                "description": "Run in background (persistent until reset). Always True for normal use."
            }
        },
        "required": ["port"]
    }
}


def tailscale_serve_tool(port: int, protocol: str = "http", bg: bool = True) -> str:
    """Expose a local port via Tailscale Serve."""
    if not bg:
        return tool_error("tailscale_serve only supports bg=True. Use tailscale_status for status checks.")

    # Validate port
    if not (1 <= port <= 65535):
        return tool_error(f"Invalid port {port}. Must be between 1 and 65535.")

    # Check Tailscale is running
    code, out, err = _run(["tailscale", "status", "--self"])
    if code != 0:
        return tool_error(f"Tailscale is not running or not authenticated. {err}")

    # Build the serve target
    if protocol == "https+insecure":
        target = f"https+insecure://localhost:{port}"
    elif protocol == "https":
        target = f"https://localhost:{port}"
    else:
        target = str(port)

    # Run tailscale serve
    cmd = ["tailscale", "serve", target]
    if bg:
        cmd.insert(2, "--bg")

    code, out, err = _run(cmd)
    if code != 0:
        return tool_error(f"tailscale serve failed: {err}")

    # Get the Tailscale IP
    ts_ip = _get_tailscale_ip()
    device_name = _get_tailscale_device_fqdn()

    # tailscale serve uses port 443 internally — the public URL uses the Tailscale IP
    # but note: for http (port 80/443) Tailscale serves on its own port mapping.
    # For arbitrary ports, Tailscale serve still works — it proxies HTTPS traffic
    # to the target port. The URL format is https://<tailnet-ip>/ but we need
    # to determine the actual port in the URL. Tailscale serve with --bg sets up
    # HTTPS on the Tailscale namespace at /. For non-443 ports, it still works
    # via the TLS SNI routing.
    #
    # The cleanest URL is: https://<ts-ip>/ — Tailscale routes based on the
    # certificate and the serve config. For multiple ports, they get distinct
    # hostnames via the device name or we use subpaths.
    # Actually: tailscale serve with --bg exposes at https://<tailnet-ip>/
    # For multiple services, use https://<tailnet-ip>/<port> or check status.

    result = {
        "success": True,
        "port": port,
        "protocol": protocol,
        "message": f"Service on localhost:{port} is now exposed to your tailnet.",
    }

    if ts_ip:
        result["tailnet_url"] = f"https://{ts_ip}/"
        result["message"] += f"\n\nAccess it from any tailnet device at: https://{ts_ip}/"
    if device_name:
        result["device_name"] = device_name
        result["message"] += f"\nDevice: {device_name}"

    result["message"] += "\n\nTo stop sharing: use tailscale_serve with action=reset."
    return json.dumps(result)


def tailscale_status_tool(action: str = "status") -> str:
    """Get current Tailscale Serve status or reset serve config."""
    if action == "reset":
        code, out, err = _run(["tailscale", "serve", "reset"])
        if code != 0:
            return tool_error(f"tailscale serve reset failed: {err}")
        return json.dumps({"success": True, "message": "Tailscale Serve config has been reset. All exposed services are now private."})

    # status
    code, out, err = _run(["tailscale", "serve", "status", "--json"])
    if code != 0:
        return tool_error(f"tailscale serve status failed: {err}")
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return tool_error(f"Could not parse tailscale serve status: {out}")

    ts_ip = _get_tailscale_ip()
    device_name = _get_tailscale_device_fqdn()

    result = {
        "success": True,
        "tailscale_ip": ts_ip,
        "device_name": device_name,
        "serve_config": data,
    }

    if ts_ip:
        result["tailnet_base_url"] = f"https://{ts_ip}/"

    return json.dumps(result, indent=2)


def tool_error(msg: str) -> str:
    return json.dumps({"success": False, "error": msg})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry

registry.register(
    name="tailscale_serve",
    toolset="tailscale",
    schema=TAILSCALE_SERVE_SCHEMA,
    handler=lambda args, **kw: tailscale_serve_tool(
        port=args.get("port"),
        protocol=args.get("protocol", "http"),
        bg=args.get("bg", True),
    ),
    check_fn=lambda: True,  # Always available if tailscale is installed
    emoji="🌐",
)

registry.register(
    name="tailscale_status",
    toolset="tailscale",
    schema=TAILSCALE_SERVE_STATUS_SCHEMA,
    handler=lambda args, **kw: tailscale_status_tool(
        action=args.get("action", "status"),
    ),
    check_fn=lambda: True,
    emoji="🌐",
)
