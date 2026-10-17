@echo off
cd /d %~dp0
where git >nul 2>nul
if errorlevel 1 (
    echo [ERROR] git command not found.
    echo Please install Git first, or open VS Code and run this in its terminal.
    echo.
    pause
    exit /b 1
)
echo ============================================
echo   Pushing RAG project to GitHub...
echo   Repo: github.com/qfl111/rag-knowledge-qa
echo ============================================
echo.
git push origin main
echo.
if errorlevel 1 (
    echo [FAILED] Push failed. Possible reasons:
    echo   1. Your proxy software is not running - turn it on and double-click this file again
    echo   2. Network issue - just try again
    echo   3. Login required - a browser window will pop up for GitHub login
) else (
    echo [SUCCESS] Pushed! GitHub repo is now up to date.
    echo Refresh this page to see the updates:
    echo https://github.com/qfl111/rag-knowledge-qa
)
echo.
pause
