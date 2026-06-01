# MCP Server Setup Guide

This guide explains how to connect the hallucination detection system to Claude via MCP.

---

## Structure

```
src/mcp/
├── mcp.types.ts        # JSON-RPC 2.0 type definitions
├── mcp.service.ts      # Protocol handler + tool implementations
├── mcp.controller.ts   # POST /mcp endpoint
└── mcp.module.ts       # NestJS module
```

MCP endpoint: `POST http://localhost:8000/mcp`

---

## Running the Server

```bash
# Development mode (auto-reload on file changes)
npm run start:dev

# Production
npm run build
node dist/main.js
```

Once the server starts, you should see:

```
Server running at: http://localhost:8000
MCP endpoint: http://localhost:8000/mcp
```

---

## Register with Claude Code

Run this once from the project root:

```bash
claude mcp add --transport http hallucination-detector http://localhost:8000/mcp
```

Verify the connection:

```bash
claude mcp list
# hallucination-detector: http://localhost:8000/mcp (HTTP) - ✓ Connected
```

> **Note**: Start a new Claude Code session after registering to activate the tools.

---

## Register with Claude Desktop

Add the following to your Claude Desktop config file:

- **Mac**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "hallucination-detector": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

Restart Claude Desktop after saving.

---

## Running with Docker

```bash
# Build the image
docker build -t hallucination-detector .

# Run the container
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  --name hallucination-detector \
  hallucination-detector
```

Or with docker-compose:

```bash
docker-compose up -d
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | No | `8000` | Server port |
| `OPENAI_API_KEY` | Yes* | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | Yes* | - | Anthropic API key |
| `HF_API_KEY` | No | - | Hugging Face Read Token (Required for custom NLI or inference api) |
| `HF_MODEL_ID` | No | `microsoft/deberta-v2-xlarge-mnli` | Hugging Face custom model path (e.g., `username/klue-roberta-small-nli`) |
| `LOG_LEVEL` | No | `info` | Log level |

*Required after LLM-based layer implementation

---

## Available Tools

### `check_hallucination`

Detects hallucinations in LLM responses.

**Input**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `question` | string | Yes | The question asked to the LLM |
| `response` | string | Yes | The LLM response to inspect |
| `context` | string | No | Additional context |

**Output**

```json
{
  "is_hallucination": true,
  "confidence": 0.95,
  "reason": "3 suspicious patterns detected",
  "flagged_parts": [
    "Overconfident expression",
    "Unverified source citation",
    "Specific date claim"
  ]
}
```

**Example (curl)**

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "check_hallucination",
      "arguments": {
        "question": "When was Admiral Yi Sun-sin born?",
        "response": "As all experts always agree, he was born in 1545."
      }
    }
  }'
```

---

## Diagnostics

Check that the server is responding:

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"ping"}'

# Expected: {"jsonrpc":"2.0","id":1,"result":{}}
```

List registered tools:

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

---

## Adding a New Tool

Edit `src/mcp/mcp.service.ts`.

**1. Add the tool definition to the `tools` array**

```typescript
{
  name: 'my_tool',
  description: 'Description of what the tool does',
  inputSchema: {
    type: 'object',
    properties: {
      input: { type: 'string', description: 'Input value' },
    },
    required: ['input'],
  },
}
```

**2. Add a branch in `handleToolCall()`**

```typescript
if (params.name === 'my_tool') {
  // your logic here
  return {
    jsonrpc: '2.0',
    id: req.id,
    result: {
      content: [{ type: 'text', text: 'result' }],
    },
  };
}
```
