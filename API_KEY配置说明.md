# DashScope API Key 配置说明

## 方法1：在启动脚本中设置（简单但不安全）

### Linux/WSL (start_backend.sh)
编辑 `start_backend.sh` 文件，取消注释并填入您的API Key：
```bash
export DASHSCOPE_API_KEY="sk-your-api-key-here"
```

### Windows (start_backend.bat)
编辑 `start_backend.bat` 文件，取消注释并填入您的API Key：
```batch
set DASHSCOPE_API_KEY=sk-your-api-key-here
```

## 方法2：设置系统环境变量（推荐，更安全）

### Linux/WSL
在 `~/.bashrc` 或 `~/.zshrc` 文件中添加：
```bash
export DASHSCOPE_API_KEY="sk-your-api-key-here"
```
然后运行：
```bash
source ~/.bashrc  # 或 source ~/.zshrc
```

### Windows
1. 右键"此电脑" -> "属性"
2. 点击"高级系统设置"
3. 点击"环境变量"
4. 在"用户变量"或"系统变量"中点击"新建"
5. 变量名：`DASHSCOPE_API_KEY`
6. 变量值：`sk-your-api-key-here`
7. 点击"确定"保存

## 方法3：在代码中直接设置（不推荐，仅用于测试）

编辑 `backend/main.py` 文件，在文件开头添加：
```python
os.environ["DASHSCOPE_API_KEY"] = "sk-your-api-key-here"
```

⚠️ **注意**：不要将包含API Key的代码提交到版本控制系统（如Git）！

## 验证配置

启动后端服务后，访问 `http://localhost:8000/api/health`，检查返回的JSON中：
- `qwen_api_available` 应该为 `true`

## 获取 API Key

1. 访问 [阿里云DashScope控制台](https://dashscope.console.aliyun.com/)
2. 登录您的账号
3. 在"API-KEY管理"中创建或查看您的API Key
4. 复制API Key（格式通常为 `sk-...`）

