# SonarCloud Setup Guide for Oil Price Prediction Backend

This guide explains how to set up and use SonarCloud for code quality analysis.

## What is SonarCloud?

SonarCloud is a cloud-based code quality and security service that automatically analyzes your code for:
- Bugs and vulnerabilities
- Code smells and maintainability issues
- Security hotspots
- Test coverage
- Code duplications

## Setup Instructions

### Step 1: SonarCloud Account Setup

1. Go to [https://sonarcloud.io](https://sonarcloud.io)
2. Sign in with your GitHub account
3. Authorize SonarCloud to access your GitHub repositories

### Step 2: Import Repository

1. Click on "+" in the top-right corner
2. Select "Analyze new project"
3. Choose your organization or create a new one
4. Select `fyp_backend` repository
5. Click "Set Up"

### Step 3: Configure GitHub Secrets

1. In SonarCloud, go to your project → Administration → Security
2. Generate a new token (or use existing one)
3. Go to GitHub repository → Settings → Secrets and variables → Actions
4. Add new repository secret:
   - Name: `SONAR_TOKEN`
   - Value: [Your SonarCloud token]

### Step 4: Verify Setup

1. Push any commit to `main` or `develop` branch
2. GitHub Actions will automatically trigger the SonarCloud workflow
3. View results at: `https://sonarcloud.io/dashboard?id=PramudithaN_fyp_backend`

## Configuration Files

### `sonar-project.properties`

Main configuration file that defines:
- Project key and organization
- Source and test directories
- Coverage report paths
- Exclusions and quality gate settings

```properties
sonar.projectKey=PramudithaN_fyp_backend
sonar.organization=pramudithan
sonar.sources=app
sonar.tests=tests
sonar.python.coverage.reportPaths=coverage.xml
```

### `.github/workflows/sonarcloud.yml`

GitHub Actions workflow that:
- Runs on push and pull requests
- Executes tests with coverage
- Sends results to SonarCloud

## Viewing Results

### SonarCloud Dashboard

Access your project dashboard at:
https://sonarcloud.io/dashboard?id=PramudithaN_fyp_backend

### Badges

Add badges to README.md:

```markdown
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=PramudithaN_fyp_backend&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=PramudithaN_fyp_backend)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=PramudithaN_fyp_backend&metric=coverage)](https://sonarcloud.io/summary/new_code?id=PramudithaN_fyp_backend)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=PramudithaN_fyp_backend&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=PramudithaN_fyp_backend)
```

## Key Metrics Explained

### Quality Gate
- Pass/Fail status based on defined thresholds
- Checks coverage, duplications, issues, security, etc.

### Coverage
- Percentage of code covered by tests
- Goal: >80% for production code

### Bugs
- Definite code errors that should be fixed
- Severity: Blocker, Critical, Major, Minor, Info

### Vulnerabilities
- Security issues in your code
- Should be addressed immediately

### Code Smells
- Maintainability issues
- Technical debt measured in minutes

### Duplications
- Percentage of duplicated code blocks
- Goal: <3%

## Running Analysis Locally

### Prerequisites
```bash
# Install SonarScanner CLI
# macOS:
brew install sonar-scanner

# Windows:
# Download from https://docs.sonarcloud.io/advanced-setup/ci-based-analysis/sonarscanner-cli/
```

### Run Analysis
```bash
# Generate coverage report first
pytest --cov=app --cov-report=xml

# Run SonarScanner
sonar-scanner \
  -Dsonar.organization=pramudithan \
  -Dsonar.projectKey=PramudithaN_fyp_backend \
  -Dsonar.sources=app \
  -Dsonar.tests=tests \
  -Dsonar.python.coverage.reportPaths=coverage.xml \
  -Dsonar.host.url=https://sonarcloud.io \
  -Dsonar.login=YOUR_SONAR_TOKEN
```

## Best Practices

1. **Fix Issues Promptly**: Address bugs and vulnerabilities as they appear
2. **Maintain Coverage**: Keep test coverage above 80%
3. **Review New Code**: Focus on issues in new/changed code
4. **Quality Gate**: Don't merge PRs that fail the quality gate
5. **Regular Monitoring**: Check dashboard regularly for trends

## Troubleshooting

### Workflow Fails
- Verify `SONAR_TOKEN` secret is set correctly
- Check GitHub Actions logs for specific errors
- Ensure coverage.xml is generated before SonarCloud scan

### No Coverage Data
- Make sure pytest runs with `--cov-report=xml`
- Verify coverage.xml path matches sonar-project.properties
- Check test execution is successful

### Authentication Errors
- Regenerate SONAR_TOKEN in SonarCloud
- Update GitHub secret with new token
- Verify organization and project key match

## Additional Resources

- [SonarCloud Documentation](https://docs.sonarcloud.io/)
- [Python Analysis Parameters](https://docs.sonarcloud.io/advanced-setup/languages/python/)
- [Quality Gates](https://docs.sonarcloud.io/improving/quality-gates/)
- [Quality Profiles](https://docs.sonarcloud.io/improving/quality-profiles/)

## Support

For issues related to:
- SonarCloud setup: Check [SonarCloud Community](https://community.sonarsource.com/)
- Project-specific questions: Open an issue in the GitHub repository
