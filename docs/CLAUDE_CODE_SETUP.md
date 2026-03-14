# Claude Code 環境設定指南

> 在新電腦上重建 Claude Code 開發環境的完整步驟

---

## 1. 安裝 Claude Code

### macOS / Linux

```bash
npm install -g @anthropic-ai/claude-code
```

### Windows

```powershell
npm install -g @anthropic-ai/claude-code
```

安裝後執行 `claude` 登入帳號。

---

## 2. 全域設定 (`~/.claude/settings.json`)

```json
{
  "model": "opus",
  "extraKnownMarketplaces": {
    "claude-reflect-marketplace": {
      "source": {
        "source": "github",
        "repo": "bayramannakov/claude-reflect"
      }
    }
  },
  "enabledPlugins": {
    "claude-reflect@claude-reflect-marketplace": true
  }
}
```

> Windows 路徑：`%USERPROFILE%\.claude\settings.json`

---

## 3. 安裝 Plugin：Claude Reflect

用於記錄 session 中的修正與學習，自動更新 CLAUDE.md。

```bash
# 在 Claude Code 對話中執行
/install-plugin claude-reflect@claude-reflect-marketplace
```

或手動安裝：

```bash
# 加入 marketplace
claude settings add extraKnownMarketplaces.claude-reflect-marketplace '{"source":{"source":"github","repo":"bayramannakov/claude-reflect"}}'

# 啟用
claude settings add enabledPlugins.claude-reflect@claude-reflect-marketplace true
```

安裝後可用的 skill：
- `/reflect` — 回顧 session 修正並更新 CLAUDE.md
- `/reflect-skills` — 發現重複模式並建立 skill

---

## 4. 安裝 SuperClaude（Slash Commands 合集）

SuperClaude 提供 30+ 個 `/sc:*` 指令（分析、實作、測試、git 等）。

### 安裝方式

```bash
# 方法 A：使用安裝腳本（推薦）
npx superclaude install

# 方法 B：手動複製
# 從 GitHub 下載 superclaude 的 commands 資料夾
# 放到 ~/.claude/commands/sc/
```

### 安裝驗證

```bash
ls ~/.claude/commands/sc/
# 應該看到 ~31 個 .md 檔案
```

### 常用指令

| 指令 | 用途 |
|------|------|
| `/sc:analyze` | 程式碼品質/安全/效能分析 |
| `/sc:implement` | 功能實作 |
| `/sc:test` | 執行測試 + 覆蓋率 |
| `/sc:git` | Git 操作 + 智能 commit message |
| `/sc:explain` | 程式碼解釋 |
| `/sc:troubleshoot` | 問題診斷 |
| `/sc:research` | 深度搜尋 |
| `/sc:help` | 列出所有指令 |

---

## 5. Status Line（狀態列）

顯示 model 名稱、context 使用率、git branch、session 時間、rate limit。

### macOS / Linux

```bash
# 1. 下載 statusline script
# 從本 repo 複製或自行建立 ~/.claude/statusline.sh
# （內容見下方附錄）

# 2. 設定權限
chmod +x ~/.claude/statusline.sh

# 3. 在 settings.json 加入
# "statusLine": {
#   "type": "command",
#   "command": "bash \"$HOME/.claude/statusline.sh\""
# }
```

### Windows

Windows 上 statusline.sh 需要 Git Bash 或 WSL 才能執行。如果沒有，可以跳過此步驟。

---

## 6. 專案級設定 (`.claude/settings.local.json`)

這個檔案已在 repo 中，pull 下來即可。包含專案特定的權限設定（允許的 bash 指令等）。

---

## 7. MCP Server：Gemini（選用）

用 Gemini 作為 sub-agent 進行平行任務。

### 前置條件

- Google Gemini API Key（設定環境變數 `GEMINI_API_KEY`）
- Python 的 `mcp` 套件

### 安裝

```bash
# 1. 建立 MCP server 目錄
mkdir -p ~/mcp-servers/gemini-mcp

# 2. 複製 server 檔案
# gemini_mcp_server.py 和 gemini_helper.py
# 放到 ~/mcp-servers/gemini-mcp/

# 3. 安裝依賴
pip install mcp google-generativeai

# 4. 設定環境變數
export GEMINI_API_KEY="your_api_key"
```

---

## 快速安裝 Checklist

```
[ ] 1. npm install -g @anthropic-ai/claude-code
[ ] 2. claude（登入）
[ ] 3. 複製 settings.json 到 ~/.claude/
[ ] 4. /install-plugin claude-reflect
[ ] 5. npx superclaude install
[ ] 6. git pull（取得 .claude/settings.local.json）
[ ] 7. （選用）設定 statusline.sh
[ ] 8. （選用）設定 Gemini MCP
```

---

## 附錄：當前已安裝版本

| 元件 | 版本 |
|------|------|
| Claude Code | 2.1.76 |
| Claude Reflect | 3.0.1 |
| SuperClaude | 4.2.0（31 個 slash commands） |
| Model 設定 | opus |
