import json
import logging

import requests
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from config import Config
from openai_adapter import (
    anthropic_to_openai_response,
    anthropic_to_openai_stream_chunk,
    init_stream_state,
    cleanup_stream_state,
    openai_to_anthropic_request,
)

logger = logging.getLogger(__name__)


def create_app():
    app = Flask(__name__)
    CORS(app)

    @app.before_request
    def check_access_key():
        """接入鉴权：校验 ACCESS_API_KEY"""
        if not Config.ACCESS_API_KEY:
            return  # 未配置则不鉴权
        if request.path == '/health':
            return  # 健康检查跳过鉴权

        auth = request.headers.get('Authorization', '')
        token = ''
        if auth.startswith('Bearer '):
            token = auth[7:]
        if not token:
            token = request.headers.get('x-api-key', '')

        if token != Config.ACCESS_API_KEY:
            logger.warning(f'[auth] rejected {request.path}')
            return jsonify({
                'error': {'message': 'Invalid API key', 'type': 'authentication_error'}
            }), 401

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'ok', 'target': Config.PROXY_TARGET_URL})

    @app.route('/v1/chat/completions', methods=['POST'])
    def chat_completions():
        """OpenAI 兼容接口 — 主路由"""
        payload = request.get_json(force=True)
        is_stream = payload.get('stream', False)
        model = payload.get('model', 'unknown')
        msg_count = len(payload.get('messages', []))
        logger.info(f'[chat] model={model} stream={is_stream} messages={msg_count}')

        # 转换请求
        anthropic_payload = openai_to_anthropic_request(payload)
        logger.debug(f'[chat] anthropic_payload: {json.dumps(anthropic_payload, ensure_ascii=False)}')

        # 准备请求头
        headers = _prepare_headers()
        headers['Content-Type'] = 'application/json'

        target_url = f'{Config.PROXY_TARGET_URL.rstrip("/")}/v1/messages'

        if is_stream:
            anthropic_payload['stream'] = True
            return _handle_stream(target_url, headers, anthropic_payload)
        else:
            anthropic_payload['stream'] = False
            return _handle_non_stream(target_url, headers, anthropic_payload)

    @app.route('/v1/messages', methods=['POST'])
    def messages_passthrough():
        """Anthropic 原生格式透传"""
        payload = request.get_json(force=True)
        model = payload.get('model', 'unknown')
        is_stream = payload.get('stream', False)
        logger.info(f'[passthrough] model={model} stream={is_stream}')

        headers = _prepare_headers()
        headers['Content-Type'] = 'application/json'

        target_url = f'{Config.PROXY_TARGET_URL.rstrip("/")}/v1/messages'
        is_stream = payload.get('stream', False)

        try:
            resp = requests.post(
                target_url,
                headers=headers,
                json=payload,
                timeout=Config.API_TIMEOUT,
                stream=is_stream,
            )

            if is_stream:
                def generate():
                    for line in resp.iter_lines():
                        if line:
                            yield line.decode('utf-8', errors='replace') + '\n\n'

                return Response(generate(), content_type='text/event-stream')
            else:
                return Response(
                    resp.content,
                    status=resp.status_code,
                    content_type=resp.headers.get('Content-Type', 'application/json'),
                )
        except requests.RequestException as e:
            logger.error(f'[passthrough] request error: {e}')
            return jsonify({'error': {'message': str(e), 'type': 'proxy_error'}}), 502

    def _handle_non_stream(target_url, headers, anthropic_payload):
        """处理非流式请求"""
        try:
            resp = requests.post(
                target_url,
                headers=headers,
                json=anthropic_payload,
                timeout=Config.API_TIMEOUT,
            )

            if resp.status_code != 200:
                logger.warning(f'[chat] upstream error {resp.status_code}')
                return Response(
                    resp.content,
                    status=resp.status_code,
                    content_type=resp.headers.get('Content-Type', 'application/json'),
                )

            anthropic_data = resp.json()
            openai_response = anthropic_to_openai_response(anthropic_data)
            usage = openai_response.get('usage', {})
            logger.info(f'[chat] done prompt={usage.get("prompt_tokens", 0)} completion={usage.get("completion_tokens", 0)}')
            return jsonify(openai_response)

        except requests.RequestException as e:
            logger.error(f'[chat] request error: {e}')
            return jsonify({'error': {'message': str(e), 'type': 'proxy_error'}}), 502

    def _handle_stream(target_url, headers, anthropic_payload):
        """处理流式请求"""
        request_id = f'chatcmpl-stream-{id(request)}'

        def generate():
            init_stream_state(request_id)
            event_type = ''
            try:
                resp = requests.post(
                    target_url,
                    headers=headers,
                    json=anthropic_payload,
                    timeout=Config.API_TIMEOUT,
                    stream=True,
                )

                if resp.status_code != 200:
                    error_body = resp.content.decode('utf-8', errors='replace')
                    logger.warning(f'[stream] upstream error {resp.status_code}: {error_body[:200]}')
                    error_chunk = json.dumps({
                        'error': {
                            'message': f'Upstream error {resp.status_code}: {error_body}',
                            'type': 'upstream_error',
                        }
                    })
                    yield f'data: {error_chunk}\n\n'
                    return

                for line in resp.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode('utf-8', errors='replace')

                    if decoded.startswith('event:'):
                        event_type = decoded[6:].strip()
                        continue

                    if decoded.startswith('data:'):
                        data_str = decoded[5:].strip()
                        if not data_str:
                            continue
                        try:
                            event_data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        chunks = anthropic_to_openai_stream_chunk(
                            event_type, event_data, request_id
                        )
                        for chunk_str in chunks:
                            yield f'data: {chunk_str}\n\n'

                yield 'data: [DONE]\n\n'

            except requests.RequestException as e:
                logger.error(f'[stream] request error: {e}')
                error_chunk = json.dumps({
                    'error': {'message': str(e), 'type': 'proxy_error'}
                })
                yield f'data: {error_chunk}\n\n'
            finally:
                cleanup_stream_state(request_id)

        return Response(
            generate(),
            content_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
            },
        )

    return app


def _prepare_headers():
    """准备请求头，注入 API Key"""
    headers = {
        'anthropic-version': '2023-06-01',
    }
    key = Config.PROXY_API_KEY
    if key.startswith('sk-'):
        headers['x-api-key'] = key
    else:
        headers['Authorization'] = f'Bearer {key}'
    return headers
