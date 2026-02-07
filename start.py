import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
)

from config import Config
from app import create_app

if __name__ == '__main__':
    app = create_app()
    print(f'Proxy service starting on 0.0.0.0:{Config.PROXY_PORT}')
    print(f'Target: {Config.PROXY_TARGET_URL}')

    from waitress import serve
    serve(
        app,
        host='0.0.0.0',
        port=Config.PROXY_PORT,
        channel_timeout=Config.API_TIMEOUT,
        send_bytes=1,  # 禁用输出缓冲，逐字节发送
    )
