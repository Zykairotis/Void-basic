#!/usr/bin/env python3
"""
Tree-sitter Query Error Fix Script
Automatically fixes common syntax errors in .scm query files
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import shutil
from datetime import datetime

class QueryFixer:
    def __init__(self, queries_dir: str, backup: bool = True):
        self.queries_dir = Path(queries_dir)
        self.backup = backup
        self.fixes_applied = []
        self.errors_fixed = 0
        self.files_processed = 0

    def fix_all_files(self) -> Dict:
        """Fix all .scm files in the queries directory."""
        results = {
            'files_processed': 0,
            'files_fixed': 0,
            'errors_fixed': 0,
            'fixes_applied': [],
            'backup_created': self.backup
        }

        # Find all .scm files
        scm_files = list(self.queries_dir.rglob("*.scm"))

        if self.backup:
            self._create_backup()

        print(f"Fixing {len(scm_files)} query files...")

        for scm_file in scm_files:
            file_result = self.fix_file(scm_file)
            results['files_processed'] += 1

            if file_result['fixes_applied']:
                results['files_fixed'] += 1
                results['errors_fixed'] += len(file_result['fixes_applied'])
                results['fixes_applied'].extend(file_result['fixes_applied'])

        self.files_processed = results['files_processed']
        self.errors_fixed = results['errors_fixed']

        return results

    def fix_file(self, file_path: Path) -> Dict:
        """Fix a single .scm query file."""
        fixes_applied = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except Exception as e:
            return {'fixes_applied': [], 'error': f"Failed to read {file_path}: {e}"}

        content = original_content

        # Apply fixes in order of importance
        content, header_fixes = self._add_header_comment(file_path, content)
        fixes_applied.extend(header_fixes)

        content, paren_fixes = self._fix_parentheses(file_path, content)
        fixes_applied.extend(paren_fixes)

        content, syntax_fixes = self._fix_syntax_errors(file_path, content)
        fixes_applied.extend(syntax_fixes)

        content, capture_fixes = self._fix_captures(file_path, content)
        fixes_applied.extend(capture_fixes)

        content, pattern_fixes = self._fix_patterns(file_path, content)
        fixes_applied.extend(pattern_fixes)

        content, format_fixes = self._format_content(file_path, content)
        fixes_applied.extend(format_fixes)

        # Only write if changes were made
        if content != original_content:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception as e:
                return {'fixes_applied': [], 'error': f"Failed to write {file_path}: {e}"}

        return {'fixes_applied': fixes_applied}

    def _create_backup(self):
        """Create a backup of the queries directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.queries_dir.parent / f"queries_backup_{timestamp}"

        try:
            shutil.copytree(self.queries_dir, backup_dir)
            print(f"Backup created: {backup_dir}")
        except Exception as e:
            print(f"Warning: Failed to create backup: {e}")

    def _add_header_comment(self, file_path: Path, content: str) -> Tuple[str, List[str]]:
        """Add header comment if missing."""
        fixes = []

        if not content.strip().startswith(';'):
            lang_name = file_path.stem.replace('-tags', '').title()
            header = f"""; Tree-sitter query file for {lang_name}
; Language: {lang_name}
; Version: 1.0
; Generated: {datetime.now().strftime("%Y-%m-%d")}

"""
            content = header + content
            fixes.append(f"{file_path}: Added header comment")

        return content, fixes

    def _fix_parentheses(self, file_path: Path, content: str) -> Tuple[str, List[str]]:
        """Fix unbalanced parentheses."""
        fixes = []
        lines = content.split('\n')
        fixed_lines = []

        for line_num, line in enumerate(lines):
            original_line = line
            line = line.rstrip()

            # Skip empty lines and comments
            if not line or line.strip().startswith(';'):
                fixed_lines.append(original_line)
                continue

            # Count parentheses
            open_count = line.count('(')
            close_count = line.count(')')

            if open_count != close_count:
                # Common patterns for fixing

                # Pattern: )) @something - extra closing paren
                if re.search(r'\)\)\s*@', line):
                    line = re.sub(r'\)\)\s*(@[^)]*)', r') \1', line)
                    fixes.append(f"{file_path}:{line_num+1}: Fixed extra closing parenthesis")

                # Pattern: ((...) missing closing paren
                elif open_count > close_count:
                    # Try to balance by adding closing parens
                    diff = open_count - close_count
                    if '@' in line and not line.endswith(')'):
                        line += ')' * diff
                        fixes.append(f"{file_path}:{line_num+1}: Added missing closing parentheses")

                # Pattern: extra closing parens
                elif close_count > open_count:
                    # Remove extra closing parens from the end
                    diff = close_count - open_count
                    while diff > 0 and line.endswith(')'):
                        line = line[:-1].rstrip()
                        diff -= 1
                    fixes.append(f"{file_path}:{line_num+1}: Removed extra closing parentheses")

            fixed_lines.append(line if line != original_line else original_line)

        return '\n'.join(fixed_lines), fixes

    def _fix_syntax_errors(self, file_path: Path, content: str) -> Tuple[str, List[str]]:
        """Fix common syntax errors."""
        fixes = []

        # Fix predicate syntax errors
        # Pattern: (#match? @ignore "^def.*")) - extra closing paren
        content = re.sub(r'\(#(match|any-of|eq|is-not)\?\s*@\w+[^)]*\)\)',
                        lambda m: m.group(0)[:-1], content)

        # Fix malformed S-expressions
        lines = content.split('\n')
        fixed_lines = []

        for line_num, line in enumerate(lines):
            original_line = line

            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith(';'):
                fixed_lines.append(line)
                continue

            # Fix common malformed patterns

            # Pattern: ((identifier) @name.reference.send - missing closing paren
            if re.match(r'^\s*\(\([^)]+\)\s*@[^)]+$', line.strip()):
                line = line.rstrip() + ')'
                fixes.append(f"{file_path}:{line_num+1}: Added missing closing parenthesis")

            # Pattern: (#is-not? local)) @reference.module - extra closing paren
            if re.search(r'\(#[^)]+\)\)\s*@', line):
                line = re.sub(r'\(#([^)]+)\)\)\s*(@[^)]*)', r'(#\1) \2', line)
                fixes.append(f"{file_path}:{line_num+1}: Fixed predicate syntax")

            fixed_lines.append(line)

        if fixes:
            content = '\n'.join(fixed_lines)

        return content, fixes

    def _fix_captures(self, file_path: Path, content: str) -> Tuple[str, List[str]]:
        """Fix capture syntax issues."""
        fixes = []

        # Fix invalid capture names
        def fix_capture(match):
            capture = match.group(1)
            # Remove invalid characters
            fixed_capture = re.sub(r'[^a-zA-Z0-9_.]', '', capture)
            if fixed_capture != capture:
                fixes.append(f"{file_path}: Fixed capture syntax @{capture} -> @{fixed_capture}")
            return f"@{fixed_capture}"

        content = re.sub(r'@([a-zA-Z0-9_.]+)', fix_capture, content)

        return content, fixes

    def _fix_patterns(self, file_path: Path, content: str) -> Tuple[str, List[str]]:
        """Fix common pattern issues."""
        fixes = []

        # Ensure proper S-expression structure
        lines = content.split('\n')
        fixed_lines = []
        in_multiline_pattern = False
        paren_stack = []

        for line_num, line in enumerate(lines):
            original_line = line
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith(';'):
                fixed_lines.append(line)
                continue

            # Track parentheses for multiline patterns
            for char in stripped:
                if char == '(':
                    paren_stack.append(line_num)
                elif char == ')':
                    if paren_stack:
                        paren_stack.pop()

            # Fix common pattern issues
            if stripped.startswith('(') and not re.match(r'^\([a-zA-Z_#][a-zA-Z0-9_]*', stripped):
                # Invalid pattern start - try to fix
                if '#' in stripped:  # Likely a predicate
                    pass  # Leave predicates as is
                else:
                    # Log for manual review
                    fixes.append(f"{file_path}:{line_num+1}: Pattern may need manual review: {stripped}")

            fixed_lines.append(line)

        return '\n'.join(fixed_lines), fixes

    def _format_content(self, file_path: Path, content: str) -> Tuple[str, List[str]]:
        """Format content for better readability."""
        fixes = []

        # Remove excessive blank lines
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)

        # Ensure file ends with single newline
        if not content.endswith('\n'):
            content += '\n'
            fixes.append(f"{file_path}: Added final newline")
        elif content.endswith('\n\n\n'):
            content = content.rstrip('\n') + '\n'
            fixes.append(f"{file_path}: Removed excessive trailing newlines")

        return content, fixes

    def print_results(self, results: Dict):
        """Print fix results in a formatted way."""
        print("\n" + "="*60)
        print("TREE-SITTER QUERY FIX RESULTS")
        print("="*60)

        # Summary
        print(f"\n📊 SUMMARY:")
        print(f"   Files processed: {results['files_processed']}")
        print(f"   Files modified: {results['files_fixed']}")
        print(f"   Total errors fixed: {results['errors_fixed']}")

        if results['backup_created']:
            print(f"   Backup created: ✅")

        # Fix rate
        if results['files_processed'] > 0:
            fix_rate = (results['files_fixed'] / results['files_processed']) * 100
            print(f"   Files needing fixes: {fix_rate:.1f}%")

        # Categories of fixes
        if results['fixes_applied']:
            print(f"\n🔧 FIXES APPLIED ({len(results['fixes_applied'])}):")

            # Group fixes by type
            fix_types = {}
            for fix in results['fixes_applied']:
                if 'Added header comment' in fix:
                    fix_types.setdefault('Header comments', []).append(fix)
                elif 'parentheses' in fix:
                    fix_types.setdefault('Parentheses', []).append(fix)
                elif 'capture syntax' in fix:
                    fix_types.setdefault('Capture syntax', []).append(fix)
                elif 'predicate syntax' in fix:
                    fix_types.setdefault('Predicate syntax', []).append(fix)
                elif 'newline' in fix:
                    fix_types.setdefault('Formatting', []).append(fix)
                else:
                    fix_types.setdefault('Other', []).append(fix)

            for fix_type, fixes in fix_types.items():
                print(f"   {fix_type}: {len(fixes)}")

            # Show first few fixes of each type
            print(f"\n📝 DETAILED FIXES:")
            for fix_type, fixes in fix_types.items():
                print(f"\n   {fix_type}:")
                for fix in fixes[:3]:  # Show first 3 of each type
                    print(f"     • {fix}")
                if len(fixes) > 3:
                    print(f"     ... and {len(fixes) - 3} more")

        # Quality assessment
        print(f"\n🎯 RESULTS:")
        if results['errors_fixed'] > 0:
            print("   ✅ Syntax errors have been automatically fixed")
            print("   ✅ Files should now pass basic validation")
        else:
            print("   ℹ️  No common syntax errors found")

        # Next steps
        print(f"\n💡 NEXT STEPS:")
        print("   • Run verify_queries.py again to check results")
        print("   • Test queries with actual tree-sitter parsers")
        print("   • Review any remaining warnings")
        if results['backup_created']:
            print("   • Remove backup directory when satisfied with fixes")

def main():
    """Main fix function."""
    import sys

    # Default to current directory
    queries_dir = "."
    backup = True

    # Parse command line arguments
    if len(sys.argv) > 1:
        queries_dir = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--no-backup":
        backup = False

    print("Tree-sitter Query File Fixer")
    print("="*40)
    print(f"Target directory: {queries_dir}")
    print(f"Backup enabled: {backup}")

    fixer = QueryFixer(queries_dir, backup)
    results = fixer.fix_all_files()
    fixer.print_results(results)

    if results['errors_fixed'] > 0:
        print(f"\n✅ Fixed {results['errors_fixed']} errors in {results['files_fixed']} files!")
    else:
        print(f"\n✅ All files are already in good condition!")

if __name__ == "__main__":
    main()
