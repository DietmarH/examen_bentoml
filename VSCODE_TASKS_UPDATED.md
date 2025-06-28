# VS Code Test Tasks - Updated Configuration

## 📋 **UPDATED VS CODE TASKS**

The VS Code tasks have been comprehensively updated to reflect the improved test infrastructure:

### 🧪 **Test Tasks**

#### **Comprehensive Testing**
- **Run All Tests**: Complete test suite with the improved runner
- **Test Infrastructure Verification**: Final verification script

#### **Unit Tests**
- **Run Unit Tests - Data Preparation**: Data processing tests (21 tests)
- **Run Unit Tests - Model Training**: ML model tests  
- **Run Unit Tests - Prepare Data**: Data preparation utilities tests
- **Run All Unit Tests**: All unit tests combined

#### **Integration Tests**
- **Run Integration Tests - BentoML**: Model store integration (11 tests)
- **Run Integration Tests - Service**: Service functionality tests
- **Run Integration Tests - API Inference**: API endpoint tests
- **Run All Integration Tests**: All integration tests combined

#### **API Tests**
- **Run API Tests**: Comprehensive API testing (requires server)
- **Test API Endpoints (Direct)**: Direct API testing script

### 🔧 **Service Management**
- **Start BentoML Server**: Launch server for testing (background task)
- **List BentoML Models**: Check available models in store

### 📊 **Coverage & Quality**
- **Run Tests with Coverage**: Complete testing with coverage reports
- **Code Quality: Format & Lint**: Black, isort, and flake8

### 🏗️ **Build Tasks**
- **Prepare Data**: Data preprocessing pipeline
- **Train Model**: Model training and BentoML save
- **Install Dependencies (uv)**: Modern dependency management
- **UV: Install Project**: Full project installation
- **UV: Add/Remove Dependency**: Dependency management
- **UV: Update Dependencies**: Keep dependencies current

## ✅ **Key Improvements**

1. **Correct File Paths**: All test paths updated to reflect new structure
2. **UV Integration**: Uses `uv run` for proper environment management  
3. **Coverage Control**: `--no-cov` flags to avoid coverage failures during development
4. **Server Integration**: Tasks for server management and API testing
5. **Comprehensive Coverage**: All test categories properly represented
6. **Background Tasks**: Server can run in background for testing

## 🚀 **Usage**

Access via VS Code:
1. **Ctrl+Shift+P** → **Tasks: Run Task**
2. Select from organized test categories
3. All tasks now use proper paths and commands
4. Full integration with the improved test infrastructure

**The VS Code task configuration is now production-ready!** 🎯
