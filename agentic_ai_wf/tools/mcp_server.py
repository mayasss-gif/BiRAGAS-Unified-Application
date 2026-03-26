from agents.mcp import MCPServerStdio

biomcp = MCPServerStdio(
    name="BioMCP",
    params={
        "command": "uv",
        "args": ["run", "--with", "biomcp-python", "biomcp", "run"],
    },
)