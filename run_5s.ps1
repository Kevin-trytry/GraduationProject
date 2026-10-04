 = Start-Process -FilePath "python" -ArgumentList "C:\GraduationProject\grove_vision_capture.py --port COM3 --raw-debug" -RedirectStandardOutput "C:\GraduationProject\raw_log.txt" -RedirectStandardError "C:\GraduationProject\raw_log.txt" -PassThru
Start-Sleep -Seconds 5
Stop-Process -Id .Id -Force
