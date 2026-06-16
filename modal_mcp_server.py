"""RoseMed Modal MCP server — remote tools for training and volumes (stateless HTTP)."""

import modal

app = modal.App("rosemed-mcp-server")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi==0.115.14",
    "fastmcp==2.10.6",
    "pydantic==2.11.10",
    "modal>=0.66.0",
)


def make_mcp_server():
    from fastmcp import FastMCP

    mcp = FastMCP("RoseMed Modal MCP")

    @mcp.tool()
    async def start_v2_training() -> str:
        """Start RoseMed v2 QLoRA fine-tuning on Modal A100-80GB."""
        train_fn = modal.Function.from_name("rosemed-v2-train", "train_v2")
        call = train_fn.spawn()
        return f"Training started. call_id={call.object_id}"

    @mcp.tool()
    async def list_rosemed_volumes() -> str:
        """List RoseMed Modal volumes (data + models)."""
        import json
        import subprocess

        out = subprocess.run(
            ["modal", "volume", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
        return json.dumps({"stdout": out.stdout, "stderr": out.stderr, "code": out.returncode})

    @mcp.tool()
    async def volume_ls(volume_name: str = "rosemed-data", path: str = "/") -> str:
        """List files in a RoseMed Modal volume."""
        import subprocess

        out = subprocess.run(
            ["modal", "volume", "ls", volume_name, path],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout or out.stderr

    @mcp.tool()
    async def rosemed_status() -> str:
        """Check if RoseMed training app is deployed and list recent apps."""
        import subprocess

        out = subprocess.run(
            ["modal", "app", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout or out.stderr

    return mcp


@app.function(image=image, secrets=[modal.Secret.from_name("rosemed-secrets", required_keys=[])])
@modal.asgi_app()
def web():
    from fastapi import FastAPI

    mcp = make_mcp_server()
    mcp_app = mcp.http_app(transport="streamable-http", stateless_http=True)
    fastapi_app = FastAPI(lifespan=mcp_app.router.lifespan_context)
    fastapi_app.mount("/", mcp_app, "mcp")
    return fastapi_app
