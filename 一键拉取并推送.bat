@echo off
cd /d %~dp0
echo ============================================
echo   Step 1/3: Fetching remote changes...
echo ============================================
git fetch origin
if errorlevel 1 (
    echo [FAILED] Cannot reach GitHub. Check your proxy and try again.
    echo.
    pause
    exit /b 1
)
echo.
echo ============================================
echo   Step 2/3: Merging remote changes...
echo ============================================
git pull origin main --rebase
if errorlevel 1 (
    echo.
    echo ============================================
    echo   [STOP] Merge conflict detected.
    echo   DO NOT panic - nothing is broken.
    echo   Please send a screenshot of THIS window
    echo   to the assistant for next steps.
    echo ============================================
    echo.
    pause
    exit /b 1
)
echo.
echo ============================================
echo   Step 3/3: Pushing to GitHub...
echo ============================================
git push origin main
echo.
if errorlevel 1 (
    echo [FAILED] Push failed. Check proxy / network and try again.
) else (
    echo [SUCCESS] Done! Your project is now fully synced.
    echo https://github.com/qfl111/rag-knowledge-qa
)
echo.
pause
