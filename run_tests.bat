@echo off
REM Test runner script for Windows

echo === Oil Price Prediction API Test Suite ===
echo.

REM Check if pytest is installed
pytest --version >nul 2>&1
if errorlevel 1 (
    echo Error: pytest is not installed
    echo Install with: pip install -r requirements-dev.txt
    exit /b 1
)

REM Parse command line arguments
if "%1"=="" goto all_tests
if "%1"=="unit" goto unit_tests
if "%1"=="integration" goto integration_tests
if "%1"=="api" goto api_tests
if "%1"=="services" goto services_tests
if "%1"=="models" goto models_tests
if "%1"=="coverage" goto coverage_tests
if "%1"=="fast" goto fast_tests
if "%1"=="clean" goto clean
if "%1"=="help" goto help
if "%1"=="-h" goto help
if "%1"=="--help" goto help

:all_tests
echo Running all tests...
pytest tests/ -v
goto end

:unit_tests
echo Running unit tests...
pytest tests/ -m unit -v
goto end

:integration_tests
echo Running integration tests...
pytest tests/test_integration.py -v
goto end

:api_tests
echo Running API tests...
pytest tests/test_api_endpoints.py -v
goto end

:services_tests
echo Running service tests...
pytest tests/test_services.py -v
goto end

:models_tests
echo Running model tests...
pytest tests/test_models.py -v
goto end

:coverage_tests
echo Running tests with coverage...
pytest --cov=app --cov-report=html --cov-report=term-missing
echo.
echo Coverage report generated in htmlcov/
goto end

:fast_tests
echo Running tests in parallel...
pytest -n auto -v
goto end

:clean
echo Cleaning test artifacts...
if exist .pytest_cache rmdir /s /q .pytest_cache
if exist htmlcov rmdir /s /q htmlcov
if exist .coverage del /q .coverage
for /d /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
echo Cleaned!
goto end

:help
echo Usage: run_tests.bat [option]
echo.
echo Options:
echo   unit         Run unit tests only
echo   integration  Run integration tests
echo   api          Run API endpoint tests
echo   services     Run service layer tests
echo   models       Run model tests
echo   coverage     Run tests with coverage report
echo   fast         Run tests in parallel
echo   clean        Clean test artifacts
echo   help         Show this help message
echo.
echo No option: Run all tests
goto end

:end
if %errorlevel% equ 0 (
    echo.
    echo All tests passed!
) else (
    echo.
    echo Some tests failed
)
exit /b %errorlevel%
