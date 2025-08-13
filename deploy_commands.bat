@echo off
echo 🚀 AI Internship Projects - Deployment Script
echo =============================================

echo.
echo Step 1: Adding modified files to git...
git add app.py usage_manager.py env_manager.py STREAMLIT_CLOUD_FIX.md STREAMLIT_DEPLOYMENT_STEPS.md Dockerfile

echo.
echo Step 2: Committing changes...
git commit -m "CRITICAL FIX: Resolve 'bool' object has no attribute 'lower' error - Fix boolean handling in usage_manager.py and env_manager.py - Add safe boolean conversion methods - Update Streamlit Cloud deployment configuration - Ready for LinkedIn deployment"

echo.
echo Step 3: Pushing to repository...
git push origin master

echo.
echo ✅ Deployment preparation complete!
echo.
echo Next steps:
echo 1. Go to https://share.streamlit.io
echo 2. Click "New app" or update existing deployment
echo 3. Select your repository and main branch
echo 4. Set main file to: app.py
echo 5. Click Deploy
echo.
pause