# 医疗模型对话界面

这是一个用于接入医疗数据微调模型的前后端应用。

## 项目结构

```
.
├── frontend/          # Vue前端应用
│   ├── src/
│   │   ├── App.vue    # 主应用组件
│   │   └── ...
│   └── package.json
├── backend/           # FastAPI后端服务
│   ├── main.py        # 后端主程序
│   ├── requirements.txt
│   └── config.txt     # 模型路径配置
└── README.md
```

## 安装步骤

### 1. 安装后端依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 安装前端依赖

```bash
cd frontend
npm install
```
### 3.下载模型文件与解压

```
模型下载地址：【快传】我给你发了 me...ft.zip, 快来看看 https://www.alipan.com/t/pTbr35Jk49m6HaZK3feT 点击链接即可保存。「阿里云盘」APP ，无需下载极速在线查看，视频原画倍速播放。
解压前端压缩包
```
## 配置模型路径

有两种方式配置模型路径：

### 方式1: 通过配置文件（推荐）

编辑 `backend/config.txt` 文件，输入您的模型路径：

```
C:/path/to/your/model
```

### 方式2: 通过API接口

启动后端服务后，访问：
```
POST http://localhost:8000/api/load_model?path=您的模型路径
```

## 运行应用

### 方式1：使用启动脚本（推荐）

**Windows系统：**
- 双击 `start_backend.bat` 启动后端
- 双击 `start_frontend.bat` 启动前端

**Linux/Mac系统：**
- 执行 `./start_backend.sh` 启动后端（需先添加执行权限：`chmod +x start_backend.sh`）
- 在新终端执行 `cd frontend && npm run dev` 启动前端

### 方式2：手动启动

**1. 启动后端服务**

```bash
cd backend
python main.py
```
或
```bash
cd backend
python3 main.py
```

后端服务将在 `http://localhost:8000` 启动

**2. 启动前端应用**

```bash
cd frontend
npm run dev
```

前端应用将在 `http://localhost:3000` 启动

## 使用说明

1. 确保模型路径正确配置
2. 启动后端服务（会自动加载模型）
3. 启动前端应用
4. 在浏览器中打开 `http://localhost:3000`
5. 开始与医疗模型对话

## API接口

所有API接口都位于 `/api` 路径下。

### 加载模型
```
POST /api/load_model?path=模型路径
```

### 聊天接口
```
POST /api/chat
Content-Type: application/json
Body: {
  "message": "用户消息",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

### 健康评估接口
```
POST /api/assess
Content-Type: application/json
Body: {
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```
根据历史对话生成健康评估报告。

### 健康检查
```
GET /api/health
```

### 根路径
```
GET /
```
返回服务状态和模型加载情况。

## 注意事项

1. 确保您的模型是使用 `transformers` 库训练的（如BERT、GPT等）
2. 如果模型路径包含中文或特殊字符，请使用绝对路径
3. 首次加载模型可能需要较长时间，请耐心等待
4. 如果使用GPU，确保已安装CUDA版本的PyTorch

## 故障排除

### 模型加载失败
- 检查模型路径是否正确
- 确认模型文件完整
- 查看后端控制台的错误信息

### 前端无法连接后端
- 确认后端服务已启动（http://localhost:8000）
- 检查CORS配置
- 查看浏览器控制台的错误信息

### 生成回复失败
- 检查模型是否已正确加载
- 查看后端日志中的错误信息
- 确认模型输入输出格式是否正确

