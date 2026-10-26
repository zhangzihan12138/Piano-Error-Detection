' 钢琴纠错 Gradio 服务开机自启脚本（隐藏窗口，不弹黑框）
'
' 安全策略：
'   1. 启动前先探活端口 7860，若已监听则直接退出（避免重复拉起多个 app.py 抢端口）
'   2. 仅在端口空闲时才启动 .venv\Scripts\python.exe app.py
'   3. 日志追加写入 app_run.log
'
' 手动验证：双击本 vbs，或在 cmd 里执行 cscript start_server_hidden.vbs

Option Explicit

Dim ws, fso, projDir, portFree
Set ws = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

projDir = fso.GetParentFolderName(WScript.ScriptFullName) & "\"

' ---- 1. 端口探活：7860 已被占用 = 已有实例在跑，直接退出 ----
portFree = ws.Run("cmd /c " & chr(34) & "powershell -NoProfile -Command " & _
    chr(34) & "$c=Get-NetTCPConnection -LocalPort 7860 -ErrorAction SilentlyContinue; if($c){exit 1}else{exit 0}" & chr(34) & chr(34), _
    0, True)

If portFree <> 0 Then
    ws.LogEvent 0, "[Piano] 端口 7860 已被占用，跳过启动。"
    WScript.Quit 0
End If

' ---- 2. 启动 app.py（隐藏窗口）----
Dim cmd
cmd = "cmd /c cd /d " & chr(34) & projDir & chr(34) & " & " & _
    chr(34) & projDir & ".venv\Scripts\python.exe" & chr(34) & " -u app.py >> app_run.log 2>&1"

ws.LogEvent 0, "[Piano] 启动 Gradio 服务: " & cmd
ws.CurrentDirectory = projDir
ws.Run cmd, 0, False
