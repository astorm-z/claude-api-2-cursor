import json
import uuid

from tool_use_fixer import (
    normalize_tool_arguments,
    repair_exact_match_tool_arguments,
    fix_tool_use_response,
)

STOP_REASON_MAP = {
    'end_turn': 'stop',
    'max_tokens': 'length',
    'tool_use': 'tool_calls',
    'stop_sequence': 'stop',
}

# 流式 tool_use 状态管理，key 为 request_id
_STREAM_TOOL_STATE = {}


def _gen_id():
    return f'chatcmpl-{uuid.uuid4().hex[:29]}'


# ─── 请求转换 ───────────────────────────────────────────────

def openai_to_anthropic_request(payload):
    """将 OpenAI 格式请求转换为 Anthropic 格式"""
    messages = payload.get('messages', [])
    anthropic_messages = []
    system_parts = []

    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')

        # system 消息提取到顶层
        if role == 'system':
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        system_parts.append(part['text'])
                    elif isinstance(part, str):
                        system_parts.append(part)
            else:
                system_parts.append(str(content))
            continue

        # 角色映射
        anthropic_role = 'assistant' if role == 'assistant' else 'user'

        # content 处理
        anthropic_content = _convert_content(msg)

        # assistant 消息中的 tool_calls → tool_use content blocks
        if role == 'assistant' and 'tool_calls' in msg:
            if isinstance(anthropic_content, str):
                blocks = []
                if anthropic_content:
                    blocks.append({'type': 'text', 'text': anthropic_content})
            elif isinstance(anthropic_content, list):
                blocks = list(anthropic_content)
            else:
                blocks = []

            for tc in msg['tool_calls']:
                func = tc.get('function', {})
                arguments = func.get('arguments', '{}')
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                blocks.append({
                    'type': 'tool_use',
                    'id': tc.get('id', f'toolu_{uuid.uuid4().hex[:24]}'),
                    'name': func.get('name', ''),
                    'input': arguments,
                })
            anthropic_content = blocks

        # tool 角色 → tool_result
        if role == 'tool':
            tool_call_id = msg.get('tool_call_id', '')
            text_content = content if isinstance(content, str) else json.dumps(content)
            anthropic_content = [{
                'type': 'tool_result',
                'tool_use_id': tool_call_id,
                'content': text_content,
            }]
            anthropic_role = 'user'

        # 跳过空 content 的消息
        if not anthropic_content or anthropic_content == [] or anthropic_content == '':
            continue

        anthropic_messages.append({
            'role': anthropic_role,
            'content': anthropic_content,
        })

    # 合并相邻同角色消息
    anthropic_messages = _merge_consecutive_roles(anthropic_messages)

    result = {
        'model': payload.get('model', 'claude-sonnet-4-20250514'),
        'messages': anthropic_messages,
        'max_tokens': max(payload.get('max_tokens') or 8192, 8192),
    }

    if system_parts:
        result['system'] = '\n\n'.join(system_parts)

    # tools 转换
    if 'tools' in payload:
        result['tools'] = _convert_tools(payload['tools'])

    # 透传参数
    for key in ('temperature', 'top_p', 'stream'):
        if key in payload:
            result[key] = payload[key]

    return result


def _convert_content(msg):
    """转换消息 content 字段"""
    content = msg.get('content', '')
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        blocks = []
        for part in content:
            if isinstance(part, str):
                blocks.append({'type': 'text', 'text': part})
            elif isinstance(part, dict):
                part_type = part.get('type', '')
                if part_type == 'text':
                    blocks.append({'type': 'text', 'text': part.get('text', '')})
                elif part_type == 'image_url':
                    url_data = part.get('image_url', {})
                    url = url_data.get('url', '') if isinstance(url_data, dict) else str(url_data)
                    if url.startswith('data:'):
                        # base64 图片
                        media_type, _, b64 = url.partition(';base64,')
                        media_type = media_type.replace('data:', '')
                        blocks.append({
                            'type': 'image',
                            'source': {
                                'type': 'base64',
                                'media_type': media_type or 'image/png',
                                'data': b64,
                            }
                        })
                    else:
                        blocks.append({
                            'type': 'image',
                            'source': {'type': 'url', 'url': url}
                        })
        return blocks
    return str(content)


def _convert_tools(tools):
    """OpenAI tools 格式 → Anthropic tools 格式，兼容 Cursor 扁平格式"""
    result = []
    for tool in tools:
        if tool.get('type') == 'function':
            # 标准 OpenAI 嵌套格式: {type: "function", function: {name, parameters}}
            func = tool['function']
            result.append({
                'name': func.get('name', ''),
                'description': func.get('description', ''),
                'input_schema': func.get('parameters', {'type': 'object', 'properties': {}}),
            })
        elif 'name' in tool and 'input_schema' in tool:
            # Cursor 扁平格式 / Anthropic 原生格式: {name, description, input_schema}
            result.append({
                'name': tool.get('name', ''),
                'description': tool.get('description', ''),
                'input_schema': tool.get('input_schema', {'type': 'object', 'properties': {}}),
            })
    return result


def _merge_consecutive_roles(messages):
    """合并相邻同角色消息"""
    if not messages:
        return messages
    merged = [messages[0]]
    for msg in messages[1:]:
        if msg['role'] == merged[-1]['role']:
            prev_content = merged[-1]['content']
            curr_content = msg['content']
            # 统一为 list 格式合并
            prev_blocks = _to_blocks(prev_content)
            curr_blocks = _to_blocks(curr_content)
            merged[-1]['content'] = prev_blocks + curr_blocks
        else:
            merged.append(msg)
    return merged


def _to_blocks(content):
    if isinstance(content, str):
        return [{'type': 'text', 'text': content}] if content else []
    if isinstance(content, list):
        return list(content)
    return [{'type': 'text', 'text': str(content)}]


# ─── 非流式响应转换 ──────────────────────────────────────────

def anthropic_to_openai_response(response_data, request_id=None):
    """将 Anthropic 响应转换为 OpenAI 格式"""
    if not request_id:
        request_id = _gen_id()

    # 先修复 tool_use 问题
    response_data = fix_tool_use_response(response_data)

    content_text = ''
    reasoning_content = ''
    tool_calls = []
    tool_call_index = 0

    for block in response_data.get('content', []):
        if not isinstance(block, dict):
            continue
        block_type = block.get('type', '')

        if block_type == 'text':
            content_text += block.get('text', '')
        elif block_type == 'thinking':
            reasoning_content += block.get('thinking', '')
        elif block_type == 'tool_use':
            args = block.get('input', {})
            if isinstance(args, dict):
                args = normalize_tool_arguments(args)
                args = repair_exact_match_tool_arguments(block.get('name', ''), args)
            args_str = json.dumps(args) if isinstance(args, dict) else str(args)

            tool_calls.append({
                'index': tool_call_index,
                'id': block.get('id', f'toolu_{uuid.uuid4().hex[:24]}'),
                'type': 'function',
                'function': {
                    'name': block.get('name', ''),
                    'arguments': args_str,
                },
            })
            tool_call_index += 1

    stop_reason = response_data.get('stop_reason', 'end_turn')
    finish_reason = STOP_REASON_MAP.get(stop_reason, 'stop')

    message = {
        'role': 'assistant',
        'content': content_text or None,
    }
    if reasoning_content:
        message['reasoning_content'] = reasoning_content
    if tool_calls:
        message['tool_calls'] = tool_calls

    usage = response_data.get('usage', {})

    return {
        'id': request_id,
        'object': 'chat.completion',
        'model': response_data.get('model', 'claude'),
        'choices': [{
            'index': 0,
            'message': message,
            'finish_reason': finish_reason,
        }],
        'usage': {
            'prompt_tokens': usage.get('input_tokens', 0),
            'completion_tokens': usage.get('output_tokens', 0),
            'total_tokens': usage.get('input_tokens', 0) + usage.get('output_tokens', 0),
        },
    }


# ─── 流式响应转换 ────────────────────────────────────────────

def init_stream_state(request_id):
    """初始化流式状态"""
    _STREAM_TOOL_STATE[request_id] = {
        'tool_index': -1,
        'tool_buf': '',
        'current_tool_id': None,
        'current_tool_name': None,
        'input_tokens': 0,
        'output_tokens': 0,
    }


def cleanup_stream_state(request_id):
    """清理流式状态"""
    _STREAM_TOOL_STATE.pop(request_id, None)


def anthropic_to_openai_stream_chunk(event_type, event_data, request_id):
    """将 Anthropic SSE 事件转换为 OpenAI 流式 chunk

    返回值: list of (chunk_json_str) 或空列表
    """
    if not request_id:
        request_id = _gen_id()

    state = _STREAM_TOOL_STATE.get(request_id, {})
    chunks = []

    if event_type == 'message_start':
        message = event_data.get('message', {})
        usage = message.get('usage', {})
        state['input_tokens'] = usage.get('input_tokens', 0)
        chunk = _make_stream_chunk(request_id, delta={'role': 'assistant', 'content': ''})
        model = message.get('model')
        if model:
            chunk['model'] = model
        chunks.append(json.dumps(chunk))

    elif event_type == 'content_block_start':
        block = event_data.get('content_block', {})
        if block.get('type') == 'tool_use':
            state['tool_index'] += 1
            state['tool_buf'] = ''
            state['current_tool_id'] = block.get('id', f'toolu_{uuid.uuid4().hex[:24]}')
            state['current_tool_name'] = block.get('name', '')
            # 发送 tool_call 的 id 和 name
            chunk = _make_stream_chunk(request_id, delta={
                'tool_calls': [{
                    'index': state['tool_index'],
                    'id': state['current_tool_id'],
                    'type': 'function',
                    'function': {
                        'name': state['current_tool_name'],
                        'arguments': '',
                    },
                }]
            })
            chunks.append(json.dumps(chunk))

    elif event_type == 'content_block_delta':
        delta = event_data.get('delta', {})
        delta_type = delta.get('type', '')

        if delta_type == 'text_delta':
            text = delta.get('text', '')
            if text:
                chunk = _make_stream_chunk(request_id, delta={'content': text})
                chunks.append(json.dumps(chunk))

        elif delta_type == 'thinking_delta':
            thinking = delta.get('thinking', '')
            if thinking:
                chunk = _make_stream_chunk(request_id, delta={'reasoning_content': thinking})
                chunks.append(json.dumps(chunk))

        elif delta_type == 'input_json_delta':
            partial = delta.get('partial_json', '')
            state['tool_buf'] += partial
            # 累积 JSON，逐块发送 arguments 片段
            if partial:
                chunk = _make_stream_chunk(request_id, delta={
                    'tool_calls': [{
                        'index': state['tool_index'],
                        'function': {'arguments': partial},
                    }]
                })
                chunks.append(json.dumps(chunk))

    elif event_type == 'message_delta':
        delta = event_data.get('delta', {})
        stop_reason = delta.get('stop_reason', '')
        finish_reason = STOP_REASON_MAP.get(stop_reason, 'stop')
        usage = event_data.get('usage', {})
        state['output_tokens'] = usage.get('output_tokens', 0)
        chunk = _make_stream_chunk(request_id, delta={}, finish_reason=finish_reason)
        chunk['usage'] = {
            'prompt_tokens': state.get('input_tokens', 0),
            'completion_tokens': state['output_tokens'],
            'total_tokens': state.get('input_tokens', 0) + state['output_tokens'],
        }
        chunks.append(json.dumps(chunk))

    elif event_type == 'message_stop':
        cleanup_stream_state(request_id)

    return chunks


def _make_stream_chunk(request_id, delta, finish_reason=None):
    choice = {
        'index': 0,
        'delta': delta,
    }
    if finish_reason:
        choice['finish_reason'] = finish_reason
    return {
        'id': request_id,
        'object': 'chat.completion.chunk',
        'model': 'claude',
        'choices': [choice],
    }
