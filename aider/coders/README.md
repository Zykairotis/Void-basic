# Aider Coders System

A comprehensive, enhanced architecture for AI-powered code editing with improved error handling, validation, and extensibility.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Enhanced Framework](#enhanced-framework)
- [Available Coders](#available-coders)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Migration Guide](#migration-guide)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## 🎯 Overview

The Aider coders system provides multiple strategies for AI-powered code editing, each optimized for different scenarios, models, and file types. The enhanced framework introduces modern design patterns, comprehensive error handling, and intelligent coder selection.

### Key Features

- **🎨 Multiple Edit Formats**: Search/replace, unified diff, whole file, and patch formats
- **🧠 Intelligent Selection**: Automatic coder selection based on context and model capabilities
- **🔒 Enhanced Security**: Comprehensive validation and security constraints
- **⚡ Performance Optimization**: Caching, streaming, and context management
- **🛡️ Robust Error Handling**: Rich error context and recovery strategies
- **📊 Metrics & Monitoring**: Performance tracking and optimization insights

## 🏗️ Architecture

### Legacy System (Backwards Compatible)

```
Coder (Base Class)
├── EditBlockCoder          # Search/replace blocks
├── EditBlockFencedCoder    # Fenced search/replace (Claude optimized)
├── UnifiedDiffCoder        # Git-style unified diffs
├── WholeFileCoder         # Complete file replacement
├── PatchCoder             # Advanced patch format
├── AskCoder              # Read-only analysis
├── ArchitectCoder        # High-level planning
├── ContextCoder          # File identification
└── HelpCoder             # Interactive help
```

### Enhanced Framework (New)

```
EnhancedCoder (Enhanced Base)
├── Strategy Pattern
│   ├── SearchReplaceStrategy
│   ├── UnifiedDiffStrategy
│   ├── WholeFileStrategy
│   ├── PatchStrategy
│   └── NoOpStrategy
├── Factory Pattern
│   ├── EditStrategyFactory
│   └── EnhancedCoderFactory
├── Configuration Management
│   ├── ConfigManager
│   ├── AiderConfig
│   └── Profile System
└── Error Handling
    ├── Exception Hierarchy
    ├── Recovery Strategies
    └── Rich Error Context
```

## 🚀 Enhanced Framework

### Core Components

#### 1. Edit Strategies (Strategy Pattern)
Different algorithms for applying code changes:

- **SearchReplaceStrategy**: Find and replace text blocks
- **UnifiedDiffStrategy**: Apply Git-style diffs  
- **WholeFileStrategy**: Replace entire files
- **PatchStrategy**: Complex multi-file operations
- **NoOpStrategy**: Read-only operations

#### 2. Factory System (Factory Pattern)
Intelligent coder selection based on:

- Model capabilities and optimization
- File size and complexity
- Task type requirements
- User preferences
- Performance characteristics

#### 3. Configuration Management
Type-safe, validated configuration with:

- **Profiles**: development, production, safe, fast
- **Environment Variables**: Automatic override support
- **Validation**: Comprehensive configuration validation
- **Migration**: Legacy config compatibility

#### 4. Error Handling
Rich exception hierarchy with:

- **Structured Context**: File paths, line numbers, suggestions
- **Recovery Strategies**: Automatic error recovery attempts
- **User-Friendly Messages**: Clear, actionable error reporting
- **Debug Information**: Comprehensive logging for troubleshooting

## 🎭 Available Coders

### Edit-Capable Coders

| Coder | Format | Best For | Model Preference | File Size |
|-------|--------|----------|------------------|-----------|
| `EditBlockCoder` | `diff` | General editing, beginners | Any | 0-100KB |
| `EditBlockFencedCoder` | `diff-fenced` | Claude models, complex edits | Claude | 5-200KB |
| `UnifiedDiffCoder` | `udiff` | Large files, precise edits | GPT-4, Claude-3 | 20-1000KB |
| `UnifiedDiffSimpleCoder` | `udiff-simple` | Simpler diff needs | GPT-3.5, Gemini | 10-500KB |
| `WholeFileCoder` | `whole` | Small files, rewrites | Any | 0-50KB |
| `PatchCoder` | `patch` | Multi-file, complex ops | GPT-4-Turbo, Claude-3-Opus | 10-500KB |

### Read-Only Coders

| Coder | Purpose | Best For |
|-------|---------|----------|
| `AskCoder` | Code analysis | Questions about code |
| `ArchitectCoder` | High-level planning | Architecture decisions |
| `ContextCoder` | File identification | Understanding dependencies |
| `HelpCoder` | User assistance | Aider documentation |

### Editor-Focused Coders

| Coder | Format | Purpose |
|-------|--------|---------|
| `EditorEditBlockCoder` | `editor-diff` | Pure editing without shell commands |
| `EditorDiffFencedCoder` | `editor-diff-fenced` | Claude-optimized editing |
| `EditorWholeFileCoder` | `editor-whole` | Whole file editing focus |

## ⚙️ Configuration

### Quick Configuration

```python
from aider.coders import create_config_for_model, get_config_manager

# Create optimized config for your model
config = create_config_for_model("gpt-4", edit_format="udiff")

# Load a specific profile
config = get_config_manager().load_config("development")
```

### Configuration Profiles

#### Development Profile
- Verbose output enabled
- Edit confirmation required
- Backup before edits
- Enhanced validation

#### Production Profile  
- Minimal output
- Auto-commits enabled
- Performance optimized
- Streaming enabled

#### Safe Profile
- Maximum validation
- Dry-run mode available
- No file deletion
- Backup required

### Environment Variables

```bash
export AIDER_MODEL="gpt-4"
export AIDER_EDIT_FORMAT="udiff"
export AIDER_VERBOSE="true"
export AIDER_CONFIRM_EDITS="true"
export OPENAI_API_KEY="your-api-key"
```

## 💻 Usage Examples

### Basic Usage (Legacy Compatible)

```python
from aider.coders import Coder

# Create coder using original method
coder = Coder.create(
    main_model=model,
    edit_format="diff",
    io=io,
    fnames={"file1.py", "file2.py"}
)

# Run a single command
coder.run("Add error handling to the parse_file function")
```

### Enhanced Usage

```python
from aider.coders import (
    create_optimal_coder,
    create_coder_for_task,
    get_coder_factory
)

# Automatic optimal selection
coder = create_optimal_coder(
    io=io,
    task_description="Refactor user authentication system",
    files=["auth.py", "models.py", "views.py"],
    model_name="gpt-4"
)

# Task-specific creation
edit_coder = create_coder_for_task("edit", io, model_name="claude-3")
analysis_coder = create_coder_for_task("ask", io, model_name="gpt-4")

# Factory-based creation with analysis
factory = get_coder_factory()
suggestion = factory.suggest_coder_for_task(
    "Add unit tests to all service classes",
    file_paths=["services/user.py", "services/auth.py"],
    model_name="gpt-4"
)
print(f"Recommended: {suggestion['recommendations'][0]['name']}")
```

### Configuration Examples

```python
from aider.coders import ConfigBuilder, get_config_manager

# Create custom configuration
config = (ConfigBuilder()
    .with_model("gpt-4", "openai")
    .with_edit_format("udiff")
    .with_ui(verbose=True, confirm_edits=True)
    .with_security(allow_file_deletion=False)
    .build(get_config_manager()))

# Create safe development setup
from aider.coders import create_safe_coder
safe_coder = create_safe_coder(io, model_name="gpt-4", dry_run=True)
```

## 🔄 Migration Guide

### Migrating from Legacy to Enhanced

#### 1. Direct Migration (Zero Changes)
The enhanced system is backwards compatible:

```python
# This still works exactly the same
coder = Coder.create(main_model=model, edit_format="diff", io=io)
```

#### 2. Gradual Migration (Recommended)
Replace creation calls with enhanced versions:

```python
# Before
coder = Coder.create(main_model=model, edit_format="diff", io=io)

# After
from aider.coders import create_optimal_coder
coder = create_optimal_coder(io, model_name=model.name, edit_format="diff")
```

#### 3. Full Migration (Best Experience)
Use new configuration and factory systems:

```python
# Before
coder = EditBlockCoder(
    main_model=model,
    edit_format="diff",
    io=io,
    fnames=files,
    auto_commits=True,
    verbose=True
)

# After
from aider.coders import create_config_for_model, EnhancedEditBlockCoder

config = create_config_for_model(
    model.name,
    edit_format="diff",
    auto_commits=True,
    verbose=True
)

coder = EnhancedEditBlockCoder.create(
    io=io,
    config=config,
    fnames=files
)
```

### Breaking Changes (Enhanced Framework Only)

⚠️ **None!** The enhanced framework maintains 100% backwards compatibility.

### New Features Available After Migration

✅ **Enhanced Error Handling**: Rich error context and recovery  
✅ **Configuration Profiles**: Predefined optimization profiles  
✅ **Intelligent Selection**: Automatic coder/format selection  
✅ **Performance Tracking**: Metrics and optimization insights  
✅ **Security Validation**: Comprehensive file and operation validation  

## 🎯 Best Practices

### Coder Selection

#### For Small Files (<10KB)
```python
# Use whole file replacement for efficiency
coder = create_coder_for_task("rewrite", io, model_name="gpt-4")
```

#### For Large Files (>100KB)
```python
# Use precise diffs to minimize token usage
coder = create_coder_for_task("precise-edit", io, model_name="gpt-4")
```

#### For Multiple Files
```python
# Use patch format for complex multi-file operations
factory = get_coder_factory()
coder = factory.create_coder(
    io=io,
    edit_format="patch",
    model_name="gpt-4-turbo"
)
```

#### For Code Analysis
```python
# Use ask coder for read-only operations
analysis_coder = create_coder_for_task("ask", io, model_name="gpt-4")
```

### Model-Specific Optimizations

#### GPT-4 / GPT-4-Turbo
- **Best Format**: `udiff` for precision
- **Strengths**: Complex diffs, large context
- **Optimal Use**: Large files, complex refactoring

#### Claude-3 / Claude-3-Opus  
- **Best Format**: `diff-fenced` 
- **Strengths**: Structured output, multiple files
- **Optimal Use**: Complex search/replace, architectural changes

#### GPT-3.5 / Gemini
- **Best Format**: `diff`
- **Strengths**: Simple operations, cost-effective
- **Optimal Use**: Small to medium files, straightforward edits

### Security Best Practices

```python
# Use safe configuration for sensitive projects
safe_coder = create_safe_coder(
    io=io,
    model_name="gpt-4",
    dry_run=True,  # Preview changes first
    backup=True,   # Backup before edits
    confirm_edits=True  # Require confirmation
)

# Restrict file access
from aider.coders import ConfigBuilder
config = (ConfigBuilder()
    .with_security(
        allowed_file_extensions=[".py", ".js"],
        blocked_directories=["secrets", "private"],
        allow_file_deletion=False,
        max_file_size_mb=5
    )
    .build(get_config_manager()))
```

### Performance Optimization

```python
# For high-performance scenarios
config = (ConfigBuilder()
    .with_performance(
        cache_prompts=True,
        enable_streaming=True,
        max_concurrent_requests=5
    )
    .build(get_config_manager()))
```

## 🔧 Troubleshooting

### Common Issues

#### "Unknown edit format" Error
```
✅ Solution: Use get_supported_formats() to see available formats
from aider.coders import EditStrategyFactory
print(EditStrategyFactory.get_supported_formats())
```

#### "Token limit exceeded" Error
```
✅ Solutions:
- Use udiff format for large files
- Reduce file count in context
- Use streaming mode
- Break complex requests into smaller parts
```

#### "Search text not found" Error
```
✅ Solutions:
- Ensure exact whitespace matching
- Add more context lines
- Use udiff format for better precision
- Check if file was modified since last view
```

#### "File not editable" Error
```
✅ Solutions:
- Check file permissions
- Verify file type is allowed
- Check security configuration
- Ensure file is not in blocked directory
```

### Debug Mode

```python
# Enable comprehensive debugging
config = load_config("development")  # Verbose mode
config.logging.level = "DEBUG"
config.logging.log_api_calls = True

coder = create_optimal_coder(io, config=config)
```

### Performance Issues

```python
# Diagnose performance
coder = create_optimal_coder(io, model_name="gpt-4")
# ... use coder ...
report = coder.get_performance_report()
print(f"Average edit time: {report.get('average_time_ms', 0):.1f}ms")
```

## 📚 API Reference

### Factory Functions

#### `create_optimal_coder(io, **kwargs) -> Coder`
Create the best coder for your needs with automatic optimization.

**Parameters:**
- `io`: Input/output handler
- `task_description`: Optional task description for optimization
- `files`: Optional list of files to work with  
- `model_name`: AI model name (default: "gpt-4")
- `edit_format`: Optional specific format
- `**kwargs`: Additional configuration

#### `create_coder_for_task(task_type, io, model_name, **kwargs) -> Coder`
Create coder optimized for specific task types.

**Task Types:**
- `"edit"`: General code editing
- `"precise-edit"`: Large files, complex changes
- `"rewrite"`: Small files, complete rewrites
- `"ask"`: Code analysis and questions
- `"architect"`: High-level planning
- `"help"`: User assistance

### Configuration Functions

#### `load_config(profile_name="default", **overrides) -> AiderConfig`
Load configuration with profile and overrides.

**Profiles:**
- `"default"`: Balanced settings
- `"development"`: Verbose, safe settings
- `"production"`: Optimized for automation
- `"safe"`: Maximum validation and confirmations
- `"fast"`: Optimized for speed

#### `create_config_for_model(model_name, edit_format=None, **kwargs) -> AiderConfig`
Create configuration optimized for specific model.

### Enhanced Error Classes

#### `AiderCoderError`
Base exception with rich context including file paths, suggestions, and error codes.

#### `ValidationError`
Configuration or input validation failures.

#### `EditOperationError`
Errors during edit application with recovery suggestions.

#### `ConfigurationError`
Configuration loading or validation issues.

### Strategy Classes

#### `EditStrategy` (Abstract Base)
Interface for all edit strategies with validation and parsing.

#### `EditStrategyFactory`
Factory for creating appropriate edit strategies.

#### `EditStrategyCoordinator`
Coordinates multiple strategies and handles complex workflows.

## 🎨 Advanced Usage

### Custom Strategies

```python
from aider.coders import EditStrategy, EditStrategyFactory

class CustomStrategy(EditStrategy):
    @property
    def edit_format(self):
        return "custom"
    
    def parse_edits(self, content):
        # Custom parsing logic
        pass
    
    def apply_edit(self, instruction, original_content):
        # Custom application logic
        pass

# Register custom strategy
EditStrategyFactory.register_strategy("custom", CustomStrategy)
```

### Custom Coders

```python
from aider.coders import register_custom_coder, CoderCapability

class CustomCoder(EnhancedCoder):
    def _init_subclass_components(self):
        # Custom initialization
        pass

# Register with factory
register_custom_coder(
    name="custom",
    coder_class=CustomCoder,
    edit_format="custom",
    capabilities=[CoderCapability.FILE_EDITING],
    best_for=["Specialized use case"],
    model_requirements=["gpt-4"]
)
```

### Batch Operations

```python
from aider.coders import EditStrategyCoordinator

coordinator = EditStrategyCoordinator()
coordinator.set_dry_run_mode(True)  # Preview changes

results = coordinator.process_edits(
    content=ai_response,
    file_contents=file_dict
)

# Review results before applying
successful, failed = coordinator.validate_results(results)
if not failed:
    coordinator.set_dry_run_mode(False)
    coordinator.process_edits(content, file_contents)
```

## 🧪 Testing and Validation

### Validate Configuration

```python
from aider.coders import ConfigValidator

issues = ConfigValidator.validate_model_config(config.model)
if issues:
    print("Configuration issues:", issues)

suggestions = ConfigValidator.get_optimization_suggestions(config)
print("Optimization suggestions:", suggestions)
```

### Test Factory Setup

```python
from aider.coders import test_factory_configuration

if test_factory_configuration():
    print("✅ Factory configuration is valid")
else:
    print("⚠️ Factory configuration has issues")
```

### Benchmark Coders

```python
factory = get_coder_factory()

test_scenarios = [
    {
        "name": "small_file_edit",
        "context": {
            "model_name": "gpt-4",
            "file_size_kb": 5,
            "required_capabilities": ["file_editing"]
        }
    }
]

results = factory.benchmark_coders(test_scenarios)
print("Benchmark results:", results)
```

## 📊 Monitoring and Metrics

### Performance Tracking

```python
# Get performance report
coder = create_optimal_coder(io, model_name="gpt-4")
# ... use coder ...
report = coder.get_performance_report()

print(f"Sessions: {report['total_sessions']}")
print(f"Files processed: {report['total_files_processed']}")
print(f"Total time: {report['total_time_ms']:.1f}ms")
```

### Strategy Performance

```python
from aider.coders import PerformanceTracker

tracker = PerformanceTracker()
# ... record metrics during usage ...

udiff_performance = tracker.get_strategy_performance("udiff")
print(f"Success rate: {udiff_performance['success_rate_percent']:.1f}%")
```

## 🤝 Contributing

### Adding New Edit Formats

1. Create strategy class inheriting from `EditStrategy`
2. Implement required abstract methods
3. Register with `EditStrategyFactory`
4. Add tests and documentation
5. Update coder profiles in factory

### Adding New Coders

1. Create coder class inheriting from `EnhancedCoder` or legacy `Coder`
2. Implement required methods
3. Create corresponding prompt class
4. Register with `EnhancedCoderFactory`
5. Add comprehensive tests

### Testing Guidelines

- Test with multiple models (GPT-4, Claude-3, etc.)
- Test edge cases (empty files, large files, binary files)
- Test error conditions and recovery
- Validate performance characteristics
- Ensure backwards compatibility

## 📝 Changelog

### Enhanced Framework (v2.0)

**Added:**
- Strategy pattern for edit operations
- Factory pattern for coder creation
- Comprehensive configuration management
- Rich error handling with context
- Performance tracking and metrics
- Security validation framework
- Intelligent coder selection
- Configuration profiles and templates

**Improved:**
- Error messages with actionable suggestions
- Model-specific optimizations
- File size and complexity handling
- Token usage optimization
- User experience and feedback

**Maintained:**
- 100% backwards compatibility
- All existing edit formats
- Legacy coder interfaces
- Existing prompt systems

## 🔗 Related Documentation

- [Edit Formats Guide](edit-formats.md)
- [Model Optimization Guide](model-optimization.md)
- [Security Configuration](security.md)
- [Performance Tuning](performance.md)
- [API Reference](api-reference.md)

## 📞 Support

For questions about the coder system:

1. **Check this documentation** for common patterns
2. **Use the help coder**: `create_coder_for_task("help", io)`
3. **Enable debug mode** for detailed error information
4. **Check performance metrics** for optimization opportunities
5. **Review configuration** with validation tools

---

**💡 Pro Tips:**

- Use `print_coder_selection_guide()` for a quick reference
- Enable verbose mode during development for better insights
- Use dry-run mode to preview changes before applying
- Leverage configuration profiles for different environments
- Monitor performance metrics to optimize your workflow