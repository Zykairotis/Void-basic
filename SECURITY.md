# Security Documentation

## Overview

This document outlines the security measures implemented in the Void-basic project, particularly focusing on API key management and protection against accidental credential exposure.

## 🔒 Security Features

### **Git Protection**
- ✅ **Automatic `.env` exclusion**: All `.env` files are automatically ignored by git
- ✅ **GitHub secret scanning**: Push protection prevents accidental key exposure
- ✅ **Environment templates**: `.env.example` provides safe configuration templates
- ✅ **Pre-commit hooks**: Additional validation before commits

### **API Key Management**
- 🔐 **Multiple provider support**: xAI, OpenAI, Anthropic integration
- 🔐 **Key rotation**: Support for multiple API keys with fallback mechanisms
- 🔐 **Environment isolation**: Development and production key separation
- 🔐 **Usage monitoring**: Built-in tracking and alerting capabilities

## 🚨 Security Incident Response

### **If API Keys Are Exposed**

#### **Immediate Actions**
1. **Remove from Git History**:
   ```bash
   git filter-branch --force --index-filter \
     'git rm --cached --ignore-unmatch .env' \
     --prune-empty --tag-name-filter cat -- --all
   ```

2. **Regenerate Exposed Keys**:
   - **xAI**: Visit [xAI Console](https://console.x.ai/) to rotate keys
   - **OpenAI**: Visit [OpenAI Platform](https://platform.openai.com/api-keys) to regenerate
   - **Anthropic**: Visit [Anthropic Console](https://console.anthropic.com/) to rotate keys

3. **Update Local Environment**:
   ```bash
   # Update your .env file with new keys
   nano .env
   ```

4. **Force Push (if necessary)**:
   ```bash
   git push origin --force
   ```

#### **Verification Steps**
```bash
# Verify no secrets in git history
git log --all --full-history -- .env

# Check current git status
git status

# Verify .env is ignored
grep .env .gitignore
```

## 🛡️ Security Best Practices

### **For Developers**
- ✅ **Never commit `.env` files** - Use `.env.example` as template
- ✅ **Use different keys** for development and production
- ✅ **Rotate keys regularly** - Especially after team changes
- ✅ **Monitor API usage** - Check provider dashboards regularly
- ✅ **Use environment variables** - For production deployments

### **For Contributors**
- ✅ **Test with placeholder keys** before using real keys
- ✅ **Check git status** before committing
- ✅ **Use pre-commit hooks** for additional validation
- ✅ **Report security issues** immediately to maintainers

### **For Deployment**
- ✅ **Use environment variables** instead of `.env` files
- ✅ **Implement key rotation** in CI/CD pipelines
- ✅ **Monitor for unusual usage** patterns
- ✅ **Use least privilege** - Only grant necessary permissions

## 🔧 Security Configuration

### **Environment Setup**
```bash
# 1. Copy template
cp .env.example .env

# 2. Add your keys (replace with actual values)
XAI_API_KEY=xai-your-actual-key-here
OPENAI_API_KEY=sk-your-actual-key-here
ANTHROPIC_API_KEY=sk-ant-your-actual-key-here

# 3. Verify configuration
grep .env .gitignore
git status  # Should not show .env
```

### **Git Configuration**
The project includes these security measures in `.gitignore`:
```
.env
.env.local
.env.*.local
```

### **GitHub Protection**
- **Secret Scanning**: Automatically detects API keys in commits
- **Push Protection**: Blocks pushes containing detected secrets
- **Branch Protection**: Prevents force pushes to main branch
- **Code Review**: Requires review for security-sensitive changes

## 🧪 Security Testing

### **Verification Commands**
```bash
# Test environment setup
python test_model_integration.py

# Check for exposed secrets
git log --all --full-history -- .env

# Verify gitignore configuration
grep .env .gitignore

# Test API connectivity
python -c "
import os
keys = ['XAI_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY']
for key in keys:
    value = os.getenv(key)
    print(f'{key}: {\"SET\" if value else \"NOT SET\"}')
"
```

### **Security Checklist**
- [ ] `.env` file exists locally but is not tracked by git
- [ ] `.env.example` contains all required variables with placeholders
- [ ] API keys are valid and functional
- [ ] No secrets in git history
- [ ] GitHub secret scanning is enabled
- [ ] Pre-commit hooks are installed and working

## 📞 Security Contacts

### **Reporting Security Issues**
- **GitHub Issues**: Use the [Issues](https://github.com/Zykairotis/Void-basic/issues) page
- **Security Labels**: Tag issues with `security` label
- **Private Reports**: For sensitive issues, contact maintainers directly

### **Emergency Contacts**
- **Repository Owner**: [Zykairotis](https://github.com/Zykairotis)
- **Security Team**: Contact through GitHub issues with `[SECURITY]` prefix

## 📋 Security Compliance

### **Standards Adherence**
- ✅ **OWASP Guidelines**: Follows web application security best practices
- ✅ **GitHub Security**: Implements GitHub's recommended security measures
- ✅ **API Security**: Proper key management and rotation procedures
- ✅ **Environment Security**: Secure handling of configuration and secrets

### **Audit Trail**
- 📝 **Commit History**: All changes tracked and auditable
- 📝 **Issue Tracking**: Security issues documented and resolved
- 📝 **Key Rotation**: Log of key changes and rotations
- 📝 **Access Control**: Repository permissions and access logs

---

## 🔄 Security Updates

This document is updated whenever:
- New security features are implemented
- Security incidents occur and lessons are learned
- Best practices evolve
- New API providers are integrated

**Last Updated**: January 5, 2025  
**Security Status**: ✅ ACTIVE AND MONITORED  
**Next Review**: Monthly security audit 