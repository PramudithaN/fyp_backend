#!/bin/bash
# Test runner script for the Oil Price Prediction API

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Oil Price Prediction API Test Suite ===${NC}\n"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed${NC}" >&2
    echo "Install with: pip install -r requirements-dev.txt" >&2
    exit 1
fi

# Parse command line arguments
case "$1" in
    "unit")
        echo -e "${GREEN}Running unit tests...${NC}"
        pytest tests/ -m unit -v
        ;;
    "integration")
        echo -e "${GREEN}Running integration tests...${NC}"
        pytest tests/test_integration.py -v
        ;;
    "api")
        echo -e "${GREEN}Running API tests...${NC}"
        pytest tests/test_api_endpoints.py -v
        ;;
    "services")
        echo -e "${GREEN}Running service tests...${NC}"
        pytest tests/test_services.py -v
        ;;
    "models")
        echo -e "${GREEN}Running model tests...${NC}"
        pytest tests/test_models.py -v
        ;;
    "coverage")
        echo -e "${GREEN}Running tests with coverage...${NC}"
        pytest --cov=app --cov-report=html --cov-report=term-missing
        echo -e "\n${GREEN}Coverage report generated in htmlcov/${NC}"
        ;;
    "fast")
        echo -e "${GREEN}Running tests in parallel...${NC}"
        pytest -n auto -v
        ;;
    "watch")
        echo -e "${GREEN}Running tests in watch mode...${NC}"
        pytest-watch
        ;;
    "clean")
        echo -e "${YELLOW}Cleaning test artifacts...${NC}"
        rm -rf .pytest_cache htmlcov .coverage
        find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
        echo -e "${GREEN}Cleaned!${NC}"
        ;;
    "help"|"-h"|"--help")
        echo "Usage: ./run_tests.sh [option]"
        echo ""
        echo "Options:"
        echo "  unit         Run unit tests only"
        echo "  integration  Run integration tests"
        echo "  api          Run API endpoint tests"
        echo "  services     Run service layer tests"
        echo "  models       Run model tests"
        echo "  coverage     Run tests with coverage report"
        echo "  fast         Run tests in parallel"
        echo "  watch        Run tests in watch mode"
        echo "  clean        Clean test artifacts"
        echo "  help         Show this help message"
        echo ""
        echo "No option: Run all tests"
        ;;
    *)
        echo -e "${GREEN}Running all tests...${NC}"
        pytest tests/ -v
        ;;
esac

exit_code=$?

if [[ $exit_code -eq 0 ]]; then
    echo -e "\n${GREEN}✓ All tests passed!${NC}"
else
    echo -e "\n${RED}✗ Some tests failed${NC}"
fi

exit $exit_code
