' 钢琴纠错 Gradio 服务开机自启脚本（隐藏窗口运行，不弹黑框）
' 逻辑：进入项目目录 → 后台启动 app.py → 日志追加写入 app_run.log
Set ws = CreateObject("WScript.Shell")
ws.Run "cmd /c cd /d ""h:\数据挖掘\考试项目1\Piano_Project"" && "".venv\Scripts\python.exe"" app.py >> app_run.log 2>&1", 0, False
