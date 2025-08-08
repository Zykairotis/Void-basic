# Data Directory

This directory contains persistent data files used by the Void-basic project, including databases, cache files, and other data storage components.

## Contents

### Database Files
- **`project_context.db`** - Main project context database storing agent state, workflow history, and system metadata
- **`demo_project_context.db`** - Demonstration context database used for testing and examples

## File Types

### SQLite Databases (.db)
- Store persistent application state
- Contain agent memory and context information
- Track workflow execution history
- Store configuration and metadata

### Data Management

#### Context Databases
The context databases maintain:
- **Agent State** - Current state and memory of various agents
- **Project Context** - Understanding of the current project structure and goals
- **Workflow History** - Record of past operations and their outcomes
- **User Preferences** - Personalized settings and configurations
- **Performance Metrics** - Historical performance and optimization data

#### Database Schema
Context databases typically include tables for:
- `agents` - Agent registration and configuration
- `workflows` - Workflow definitions and execution logs
- `context_items` - Contextual information and relationships
- `sessions` - User session data and history
- `metadata` - System metadata and versioning

## Usage

### Backup
Regular backups are recommended:
```bash
cp data/project_context.db data/project_context_backup_$(date +%Y%m%d).db
```

### Reset
To reset application state:
```bash
rm data/project_context.db
# The application will create a new database on next run
```

### Migration
When upgrading versions:
1. Backup existing databases
2. Check for migration scripts in `scripts/` directory
3. Run any required database migrations

## Development

### Local Development
- Use `demo_project_context.db` for testing
- Avoid committing personal `project_context.db` files
- Keep database files in `.gitignore` for privacy

### Testing
- Tests should use temporary databases
- Clean up test databases after test completion
- Use fixtures for consistent test data

## Security

### Sensitive Data
Database files may contain:
- API keys and credentials
- Personal project information
- Chat history and conversations
- File system paths

### Best Practices
- Keep database files out of version control
- Use appropriate file permissions (600)
- Regular cleanup of old/unused databases
- Encryption for sensitive deployments

## Troubleshooting

### Common Issues

#### Database Locked
```
sqlite3.OperationalError: database is locked
```
- Close all applications using the database
- Check for stale lock files
- Restart the application

#### Corruption
If database corruption occurs:
1. Stop the application
2. Restore from backup
3. Check disk space and file system health
4. Consider database repair tools if needed

#### Performance
For large databases:
- Regular VACUUM operations
- Index optimization
- Consider archiving old data

## Maintenance

### Regular Tasks
- Monitor database file sizes
- Clean up old demo databases
- Backup important context data
- Optimize database performance

### Monitoring
Watch for:
- Rapid database growth
- Performance degradation
- Lock contention
- Corruption indicators

## Contributing

When working with data files:
1. Never commit personal database files
2. Provide sample/demo databases for testing
3. Document any schema changes
4. Include migration scripts for breaking changes
5. Test with both empty and populated databases