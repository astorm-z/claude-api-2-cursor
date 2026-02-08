# Claude API 2 Cursor

将 Cursor 的 OpenAI 格式请求转换为 Anthropic 格式，通过 Claude 中转站 API 实现 Cursor 使用 Claude 模型。

## 功能

- OpenAI ↔ Anthropic 请求/响应格式双向转换
- 流式 SSE 输出
- Tool Use 支持（兼容 Cursor 扁平格式和 OpenAI 标准格式）
- Tool Use 智能修复（引号容错、字段映射）
- 接入鉴权（ACCESS_API_KEY）
- Anthropic 原生格式透传（`/v1/messages`）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`：

```
PROXY_TARGET_URL=https://your-relay-station.com
PROXY_API_KEY=sk-xxx
PROXY_PORT=3029
ACCESS_API_KEY=your-access-key
```

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PROXY_TARGET_URL` | Claude 中转站地址 | `https://api.anthropic.com` |
| `PROXY_API_KEY` | 中转站 API Key | - |
| `PROXY_PORT` | 服务监听端口 | `3029` |
| `API_TIMEOUT` | 请求超时（秒） | `300` |
| `ACCESS_API_KEY` | 接入鉴权 Key（为空则不鉴权） | - |

### 3. 启动服务

```bash
python start.py
```

### 4. Cursor 配置

在 Cursor 设置中：
- **Base URL**: `http://localhost:3029`（或你的服务器地址）
- **API Key**: 填写 `ACCESS_API_KEY` 的值

## Docker 部署

### 1. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 填写必要的配置项（参考上方环境变量表）。

### 2. 启动服务

```bash
docker compose up -d
```

### 3. 查看日志

```bash
docker compose logs -f
```

### 4. 停止服务

```bash
docker compose down
```

> 服务默认监听 `PROXY_PORT`（默认 3029），支持通过 `.env` 自定义端口。容器以非 root 用户运行，内置健康检查。

## API 路由

| 路由 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | OpenAI 兼容接口（主路由） |
| `/v1/messages` | POST | Anthropic 原生格式透传 |
| `/health` | GET | 健康检查 |

## API Key 注入逻辑

服务会根据 `PROXY_API_KEY` 的前缀自动选择注入方式：
- `sk-` 开头 → `x-api-key` 请求头
- 其他 → `Authorization: Bearer` 请求头
