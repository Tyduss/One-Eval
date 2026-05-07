# Z-Eval 部署指南

> 本文档补充官方 README，解决实际部署中遇到的常见问题。

## 1. 环境要求

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Python | **3.10 - 3.12** | ⚠️ **不支持 Python 3.14+**，pydantic v1 和 langchain 不兼容 |
| Node.js | 18+ | 前端依赖 |
| npm/pnpm | 最新版 | 包管理器 |

## 2. 后端部署

### 2.1 创建 Python 环境（推荐 Conda）

```bash
# 如果是首次使用 conda，需要先接受服务条款
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 创建 Python 3.12 环境
conda create -n one-eval python=3.12 -y
conda activate one-eval
```

### 2.2 安装项目依赖

```bash
cd /path/to/One-Eval
pip install -e .
```

### 2.3 验证安装

```bash
# 确认 Python 路径正确（应该是 conda 环境路径）
which python3
# 预期输出: /opt/homebrew/Caskroom/miniconda/base/envs/one-eval/bin/python3
# ❌ 错误输出: /Library/Frameworks/Python.framework/Versions/3.14/bin/python3

# 验证 open-dataflow 是否正确安装
python -c "from dataflow.operators.core_text import BenchAnswerGenerator; print('OK')"
# 预期输出: OK（可能有一些 INFO 日志）

# 如果报错 ImportError，升级 open-dataflow
pip install --upgrade open-dataflow
```

### 2.4 启动后端服务

```bash
# ⚠️ 重要：使用 python -m 而不是直接调用 uvicorn
# 这样可以确保使用当前 conda 环境的 Python
python -m uvicorn one_eval.server.app:app --host 127.0.0.1 --port 8000

# ❌ 错误方式（可能调用到系统 Python 的 uvicorn）
# uvicorn one_eval.server.app:app --host 127.0.0.1 --port 8000
```

### 2.5 验证后端启动

```bash
# 访问 API 文档
open http://127.0.0.1:8000/docs
```

## 3. 前端部署

```bash
cd ~/One-Eval/one-eval-web
npm install
npm run dev
```

访问 http://localhost:5173

## 4. 常见问题排查

### Q1: `ImportError: cannot import name 'BenchAnswerGenerator'`

**原因**: `open-dataflow` 版本过低或安装到了错误的 Python 环境

**解决**:
```bash
# 检查安装位置
pip show open-dataflow

# Location 应该是 conda 环境路径，例如：
# /opt/homebrew/Caskroom/miniconda/base/envs/one-eval/lib/python3.12/site-packages

# 如果是其他路径（如 /Library/Frameworks/...），说明装错了环境
# 重新安装：
conda activate one-eval
pip install -e . --force-reinstall
```

### Q2: `Core Pydantic V1 functionality isn't compatible with Python 3.14`

**原因**: Python 版本过高（3.14+）

**解决**: 使用 Python 3.12 或更低版本创建 conda 环境

### Q3: 启动后端时找不到模块

**原因**: 依赖装到了系统 Python 而不是 conda 环境

**解决**:
```bash
# 确认当前 Python
which python3

# 如果不是 conda 环境路径，重新激活
conda activate one-eval

# 使用 python -m 方式启动
python -m uvicorn one_eval.server.app:app --host 127.0.0.1 --port 8000
```

### Q4: conda 创建环境时报 TOS 错误

**解决**:
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

## 5. 完整启动流程

```bash
# 1. 激活环境
conda activate one-eval

# 2. 启动后端（终端 1）
cd /path/to/One-Eval
python -m uvicorn one_eval.server.app:app --host 127.0.0.1 --port 8000

# 3. 启动前端（终端 2）
cd /path/to/One-Eval/one-eval-web
npm run dev

# 4. 访问 http://localhost:5173
# 5. 首次使用请先进入设置页面配置 API Key、模型等参数
```

## 6. 功能说明

- **任务状态持久化**: 切换 Tab 不会丢失任务进度，返回工作台后会自动恢复轮询
