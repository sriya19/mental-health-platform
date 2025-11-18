@echo off
echo ========================================
echo Starting Mental Health Platform UI
echo ========================================
echo.
echo The Streamlit UI will open in your browser automatically.
echo.
echo Features Available:
echo - Search datasets with user stories
echo - Ask AI-powered questions about Baltimore data
echo - View analytics and visualizations
echo - Manage indexed datasets
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

cd ui_app
streamlit run streamlit_app.py
