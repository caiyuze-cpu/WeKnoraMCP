#!/usr/bin/env python3
"""WeKnora MCP Server v2 - 智能体对话集成

核心功能：
- 首次对话自动识别可用智能体，引导用户选择
- 记住选择，后续对话直接使用
- 支持"切换智能体"
"""

import asyncio
import json
import logging
import os
from typing import Optional

import httpx
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

BASE_URL = os.getenv("WEKNORA_BASE_URL", "http://localhost:8080/api/v1")
API_KEY = os.getenv("WEKNORA_API_KEY", "")

# 服务端状态（进程内持久化）
_state = {
    "agent": None,       # {"id", "name", "description", "knowledge_bases"}
    "session_id": None,
}


def _headers() -> dict:
    return {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json",
    }


# ---------- 智能体管理 ----------

async def _fetch_agents() -> list:
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{BASE_URL}/agents", headers=_headers())
        if r.status_code != 200:
            raise RuntimeError(f"获取智能体失败: HTTP {r.status_code}")
        data = r.json()
        agents = data if isinstance(data, list) else data.get("data", data.get("agents", []))
        return agents if isinstance(agents, list) else []


async def _fetch_agent(agent_id: str) -> Optional[dict]:
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{BASE_URL}/agents/{agent_id}", headers=_headers())
        if r.status_code != 200:
            return None
        data = r.json()
        a = data if isinstance(data, dict) and "id" in data else data.get("data", {})
        return a


def _format_agent_list(agents: list) -> str:
    if not agents:
        return "没有可用的智能体"
    lines = []
    for a in agents:
        lines.append(f"- **{a.get('name', '?')}**  ID: `{a.get('id', '?')}`\n  {a.get('description', '')}")
    return "\n\n".join(lines)


# ---------- 会话管理 ----------

async def _create_session() -> str:
    agent_name = _state["agent"]["name"] if _state["agent"] else "default"
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            f"{BASE_URL}/sessions",
            headers=_headers(),
            json={"name": f"cc-{agent_name}"},
        )
        if r.status_code in (200, 201):
            data = r.json()
            return data.get("id") or data.get("data", {}).get("id")
        raise RuntimeError(f"创建会话失败: {r.status_code} {r.text[:200]}")


# ---------- SSE 流式对话 ----------

async def _do_agent_chat(message: str) -> str:
    payload = {
        "query": message,
        "agent_id": _state["agent"]["id"],
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=30)) as c:
        async with c.stream(
            "POST",
            f"{BASE_URL}/agent-chat/{_state['session_id']}",
            headers=_headers(),
            json=payload,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                return f"对话失败: HTTP {resp.status_code} {body.decode()[:500]}"

            answer_parts = []
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                payload_str = line[5:].strip()
                if not payload_str or payload_str == "[DONE]":
                    continue
                try:
                    event = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue

                rtype = event.get("response_type", "")
                content = event.get("content", "")
                if rtype == "answer" and content:
                    answer_parts.append(content)
                elif rtype in ("thinking", "thought") and content:
                    pass  # 跳过思考过程，只返回最终回答
                elif rtype == "error" and content:
                    return f"智能体返回错误: {content}"

            return "".join(answer_parts) if answer_parts else "（智能体未返回有效内容）"


# ---------- 知识库搜索 ----------

async def _knowledge_search(query: str, kb_ids: list) -> str:
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(
            f"{BASE_URL}/knowledge-search",
            headers=_headers(),
            json={"query": query, "knowledge_base_ids": kb_ids},
        )
        if r.status_code != 200:
            return f"搜索失败: HTTP {r.status_code} {r.text[:200]}"
        data = r.json()
        return json.dumps(data, indent=2, ensure_ascii=False)


# =====================================================
#  MCP 工具定义
# =====================================================

app = Server("weknora-server")


@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="weknora_chat",
            description=(
                "与WeKnora智能体对话。首次使用会提示选择智能体，之后自动记住选择。"
                "用户可以说'切换智能体'来重新选择，说'新建对话'来开启新的会话。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "要发送给智能体的问题或消息"},
                    "new_session": {"type": "boolean", "description": "是否新建对话（清空上下文重新开始），用户说'新建对话'时设为true"},
                },
                "required": ["message"],
            },
        ),
        types.Tool(
            name="weknora_select_agent",
            description="选择或切换WeKnora智能体。传入智能体ID完成选择。",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "智能体ID（从 weknora_list_agents 获取）"},
                },
                "required": ["agent_id"],
            },
        ),
        types.Tool(
            name="weknora_list_agents",
            description="列出所有可用的WeKnora智能体，返回名称、ID和描述。",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="weknora_current_agent",
            description="查看当前选中的智能体信息。",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    args = arguments or {}

    try:
        if name == "weknora_chat":
            return await _handle_chat(args)
        elif name == "weknora_select_agent":
            return await _handle_select_agent(args)
        elif name == "weknora_list_agents":
            return await _handle_list_agents()
        elif name == "weknora_current_agent":
            return await _handle_current_agent()
        else:
            return [types.TextContent(type="text", text=f"未知工具: {name}")]
    except Exception as e:
        logger.error(f"工具执行失败 [{name}]: {e}")
        return [types.TextContent(type="text", text=f"错误: {e}")]


# ---------- 工具实现 ----------

async def _handle_chat(args: dict) -> list:
    global _state
    message = args["message"]
    new_session = args.get("new_session", False)

    # 未选择智能体 → 列出可用智能体引导选择
    if not _state["agent"]:
        try:
            agents = await _fetch_agents()
        except Exception as e:
            return [types.TextContent(type="text", text=f"连接WeKnora失败: {e}\n请确认WeKnora服务正在运行 (http://localhost:8080)")]

        if not agents:
            return [types.TextContent(type="text", text="WeKnora中没有可用的智能体，请先在WeKnora中创建智能体。")]

        agent_list = _format_agent_list(agents)
        return [types.TextContent(
            type="text",
            text=(
                f"尚未选择智能体。请告诉我你想使用哪个智能体：\n\n"
                f"{agent_list}\n\n"
                f"请回复智能体名称或ID，我会帮你选择。"
            ),
        )]

    # 新建对话 → 重置会话
    if new_session or not _state["session_id"]:
        _state["session_id"] = await _create_session()
        if new_session and message in ("新建对话", "新对话", "new chat", ""):
            return [types.TextContent(type="text", text=f"已开启新对话（智能体: {_state['agent']['name']}），请开始提问。")]

    # 执行对话
    answer = await _do_agent_chat(message)
    return [types.TextContent(type="text", text=answer)]


async def _handle_select_agent(args: dict) -> list:
    global _state
    agent_id = args["agent_id"]

    agent = await _fetch_agent(agent_id)
    if not agent:
        return [types.TextContent(type="text", text=f"智能体 {agent_id} 不存在或无法访问")]

    _state["agent"] = {
        "id": agent.get("id", agent_id),
        "name": agent.get("name", "Unknown"),
        "description": agent.get("description", ""),
        "knowledge_bases": agent.get("knowledge_bases", []),
    }
    _state["session_id"] = None  # 重置会话

    kb_info = ""
    kbs = _state["agent"].get("knowledge_bases", [])
    if kbs:
        kb_names = [kb.get("name", kb.get("id", "?")) for kb in kbs]
        kb_info = f"\n关联知识库: {', '.join(kb_names)}"

    return [types.TextContent(
        type="text",
        text=f"已选择智能体: {_state['agent']['name']} (ID: {agent_id}){kb_info}\n现在可以开始对话了。",
    )]


async def _handle_list_agents() -> list:
    try:
        agents = await _fetch_agents()
    except Exception as e:
        return [types.TextContent(type="text", text=f"获取智能体列表失败: {e}")]

    return [types.TextContent(type="text", text=_format_agent_list(agents))]


async def _handle_current_agent() -> list:
    if _state["agent"]:
        a = _state["agent"]
        text = f"当前智能体: **{a['name']}** (ID: `{a['id']}`)\n描述: {a.get('description', '无')}"
        if a.get("knowledge_bases"):
            kb_names = [kb.get("name", "?") for kb in a["knowledge_bases"]]
            text += f"\n关联知识库: {', '.join(kb_names)}"
        return [types.TextContent(type="text", text=text)]
    return [types.TextContent(type="text", text="尚未选择智能体。使用 weknora_chat 开始对话会自动引导选择。")]


# ---------- 启动 ----------

async def run():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="weknora-server",
                server_version="2.0.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
