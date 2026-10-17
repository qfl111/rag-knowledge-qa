@echo off
cd /d %~dp0
echo ============================================
echo   Auto-fixing merge conflicts (all files)...
echo ============================================

echo [1/4] Cancelling the stuck rebase...
git rebase --abort 2>nul

echo [2/4] Merging remote updates...
git merge origin/main
if errorlevel 1 (
    echo [3/4] Conflicts found. Backing up remote README...
    git show origin/main:README.md > README-remote-backup.md 2>nul
    echo [3/4] Keeping LOCAL version of every conflicting file...
    for /f "delims=" %%i in ('git diff --name-only --diff-filter=U') do (
        echo   - keep local: %%i
        git checkout --ours "%%i" 2>nul
    )
    git add -A
    git commit -m "Merge remote updates, keep local versions"
    if errorlevel 1 (
        echo.
        echo [ERROR] Auto-fix failed. Please send a screenshot of this window.
        echo.
        pause
        exit /b 1
    )
) else (
    echo [3/4] Merged cleanly, no conflict.
)

echo [4/4] Pushing to GitHub...
git push origin main
echo.
if errorlevel 1 (
    echo [FAILED] Push failed. Please send a screenshot of this window.
) else (
    echo [SUCCESS] All done! Project fully synced.
    echo https://github.com/qfl111/rag-knowledge-qa
    echo.
    echo Remote README backup saved as: README-remote-backup.md
)
echo.
pause
