#!/usr/bin/env python3
"""
Tree-sitter Query File Verification Script
Validates syntax, patterns, and quality of .scm query files
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

class QueryVerifier:
    def __init__(self, queries_dir: str):
        self.queries_dir = Path(queries_dir)
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(int)

        # Common tree-sitter patterns
        self.valid_capture_categories = {
            'definition', 'reference', 'name', 'local', 'scope'
        }

        self.standard_definition_types = {
            'function', 'method', 'class', 'interface', 'module', 'variable',
            'constant', 'type', 'enum', 'struct', 'field', 'property',
            'parameter', 'label', 'macro', 'namespace', 'package'
        }

        self.standard_reference_types = {
            'call', 'class', 'type', 'variable', 'module', 'field',
            'property', 'interface', 'enum', 'constant', 'macro',
            'namespace', 'package', 'builtin', 'operator'
        }

    def verify_all_files(self) -> Dict:
        """Verify all .scm files in the queries directory."""
        results = {
            'total_files': 0,
            'valid_files': 0,
            'files_with_errors': 0,
            'files_with_warnings': 0,
            'languages_covered': set(),
            'errors': [],
            'warnings': [],
            'statistics': {}
        }

        # Find all .scm files
        scm_files = list(self.queries_dir.rglob("*.scm"))
        results['total_files'] = len(scm_files)

        print(f"Verifying {len(scm_files)} query files...")

        for scm_file in scm_files:
            file_result = self.verify_file(scm_file)

            # Extract language name
            lang_name = scm_file.stem.replace('-tags', '')
            results['languages_covered'].add(lang_name)

            if file_result['errors']:
                results['files_with_errors'] += 1
                results['errors'].extend(file_result['errors'])

            if file_result['warnings']:
                results['files_with_warnings'] += 1
                results['warnings'].extend(file_result['warnings'])

            if not file_result['errors']:
                results['valid_files'] += 1

        results['languages_covered'] = sorted(list(results['languages_covered']))
        results['statistics'] = dict(self.stats)

        return results

    def verify_file(self, file_path: Path) -> Dict:
        """Verify a single .scm query file."""
        file_errors = []
        file_warnings = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            file_errors.append(f"{file_path}: Failed to read file - {e}")
            return {'errors': file_errors, 'warnings': file_warnings}

        # Basic syntax checks
        syntax_result = self._check_syntax(file_path, content)
        file_errors.extend(syntax_result['errors'])
        file_warnings.extend(syntax_result['warnings'])

        # Pattern validation
        pattern_result = self._validate_patterns(file_path, content)
        file_errors.extend(pattern_result['errors'])
        file_warnings.extend(pattern_result['warnings'])

        # Completeness check
        completeness_result = self._check_completeness(file_path, content)
        file_warnings.extend(completeness_result['warnings'])

        # Update statistics
        self._update_stats(content)

        return {'errors': file_errors, 'warnings': file_warnings}

    def _check_syntax(self, file_path: Path, content: str) -> Dict:
        """Check basic S-expression syntax."""
        errors = []
        warnings = []

        lines = content.split('\n')

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith(';'):
                continue

            # Check parentheses balance
            if line.count('(') != line.count(')'):
                # Only flag if it's a complete statement (not multi-line)
                if not line.endswith('\\') and ')' in line:
                    warnings.append(f"{file_path}:{line_num}: Unbalanced parentheses: {line}")

            # Check for common syntax errors
            if line.startswith('(') and not re.match(r'^\([a-zA-Z_][a-zA-Z0-9_]*', line):
                if not line.startswith('(;'):  # Allow commented patterns
                    errors.append(f"{file_path}:{line_num}: Invalid pattern start: {line}")

            # Check capture syntax
            capture_matches = re.findall(r'@[a-zA-Z0-9_.]+', line)
            for capture in capture_matches:
                if not re.match(r'^@[a-zA-Z_][a-zA-Z0-9_.]*$', capture):
                    errors.append(f"{file_path}:{line_num}: Invalid capture syntax: {capture}")

        return {'errors': errors, 'warnings': warnings}

    def _validate_patterns(self, file_path: Path, content: str) -> Dict:
        """Validate tree-sitter query patterns."""
        errors = []
        warnings = []

        # Extract all capture names
        captures = re.findall(r'@([a-zA-Z0-9_.]+)', content)

        for capture in captures:
            parts = capture.split('.')

            # Validate capture category
            if len(parts) < 2:
                warnings.append(f"{file_path}: Capture missing category: @{capture}")
                continue

            category = parts[0]
            if category not in self.valid_capture_categories:
                warnings.append(f"{file_path}: Non-standard capture category: @{capture}")

            # Validate definition/reference types
            if category == 'definition' and len(parts) >= 3:
                def_type = parts[2]
                if def_type not in self.standard_definition_types:
                    warnings.append(f"{file_path}: Non-standard definition type: @{capture}")

            if category == 'reference' and len(parts) >= 3:
                ref_type = parts[2]
                if ref_type not in self.standard_reference_types:
                    warnings.append(f"{file_path}: Non-standard reference type: @{capture}")

        # Check for required patterns
        has_definitions = any('@definition.' in content or '@name.definition.' in content for _ in [1])
        has_references = any('@reference.' in content or '@name.reference.' in content for _ in [1])

        if not has_definitions:
            warnings.append(f"{file_path}: No definition patterns found")

        if not has_references:
            warnings.append(f"{file_path}: No reference patterns found")

        return {'errors': errors, 'warnings': warnings}

    def _check_completeness(self, file_path: Path, content: str) -> Dict:
        """Check query completeness and coverage."""
        warnings = []

        # Check for language header comment
        if not content.strip().startswith(';'):
            warnings.append(f"{file_path}: Missing header comment")

        # Check for basic language constructs
        constructs = {
            'functions': ['function_declaration', 'function_definition', 'method_declaration'],
            'variables': ['variable_declaration', 'assignment'],
            'types': ['class_declaration', 'type_declaration', 'struct_declaration'],
            'calls': ['call_expression', 'function_call']
        }

        missing_constructs = []
        for construct_type, patterns in constructs.items():
            if not any(pattern in content for pattern in patterns):
                missing_constructs.append(construct_type)

        if missing_constructs:
            warnings.append(f"{file_path}: Possibly missing constructs: {', '.join(missing_constructs)}")

        return {'warnings': warnings}

    def _update_stats(self, content: str):
        """Update verification statistics."""
        self.stats['total_lines'] += len(content.split('\n'))
        self.stats['total_patterns'] += content.count('(')
        self.stats['total_captures'] += len(re.findall(r'@[a-zA-Z0-9_.]+', content))

        # Count different capture types
        captures = re.findall(r'@([a-zA-Z0-9_.]+)', content)
        for capture in captures:
            parts = capture.split('.')
            if len(parts) >= 1:
                self.stats[f'captures_{parts[0]}'] += 1

    def print_results(self, results: Dict):
        """Print verification results in a formatted way."""
        print("\n" + "="*60)
        print("TREE-SITTER QUERY VERIFICATION RESULTS")
        print("="*60)

        # Summary
        print(f"\n📊 SUMMARY:")
        print(f"   Total files: {results['total_files']}")
        print(f"   Valid files: {results['valid_files']}")
        print(f"   Files with errors: {results['files_with_errors']}")
        print(f"   Files with warnings: {results['files_with_warnings']}")
        print(f"   Languages covered: {len(results['languages_covered'])}")

        # Success rate
        if results['total_files'] > 0:
            success_rate = (results['valid_files'] / results['total_files']) * 100
            print(f"   Success rate: {success_rate:.1f}%")

        # Languages
        print(f"\n🌍 LANGUAGES COVERED ({len(results['languages_covered'])}):")
        for i, lang in enumerate(results['languages_covered']):
            if i % 8 == 0:
                print(f"\n   ", end="")
            print(f"{lang:<12}", end="")
        print()

        # Statistics
        if results['statistics']:
            print(f"\n📈 STATISTICS:")
            stats = results['statistics']
            print(f"   Total lines: {stats.get('total_lines', 0):,}")
            print(f"   Total patterns: {stats.get('total_patterns', 0):,}")
            print(f"   Total captures: {stats.get('total_captures', 0):,}")
            print(f"   Definition captures: {stats.get('captures_definition', 0):,}")
            print(f"   Reference captures: {stats.get('captures_reference', 0):,}")
            print(f"   Name captures: {stats.get('captures_name', 0):,}")

        # Errors
        if results['errors']:
            print(f"\n❌ ERRORS ({len(results['errors'])}):")
            for error in results['errors'][:10]:  # Show first 10
                print(f"   {error}")
            if len(results['errors']) > 10:
                print(f"   ... and {len(results['errors']) - 10} more")

        # Warnings
        if results['warnings']:
            print(f"\n⚠️  WARNINGS ({len(results['warnings'])}):")
            for warning in results['warnings'][:10]:  # Show first 10
                print(f"   {warning}")
            if len(results['warnings']) > 10:
                print(f"   ... and {len(results['warnings']) - 10} more")

        # Quality assessment
        print(f"\n🎯 QUALITY ASSESSMENT:")
        if results['files_with_errors'] == 0:
            print("   ✅ All files have valid syntax")
        else:
            print(f"   ❌ {results['files_with_errors']} files have syntax errors")

        if results['files_with_warnings'] < results['total_files'] * 0.1:
            print("   ✅ Very few warnings detected")
        elif results['files_with_warnings'] < results['total_files'] * 0.3:
            print("   ⚠️  Some warnings detected")
        else:
            print("   ❌ Many warnings detected - review recommended")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if results['files_with_errors'] > 0:
            print("   • Fix syntax errors before deployment")
        if results['files_with_warnings'] > 0:
            print("   • Review warnings for potential improvements")
        if len(results['languages_covered']) < 100:
            print("   • Consider adding more language support")
        print("   • Test queries with real code samples")
        print("   • Keep queries updated with parser versions")

def main():
    """Main verification function."""
    import sys

    # Default to current directory's queries folder
    queries_dir = "."

    # Override if provided as command line argument
    if len(sys.argv) > 1:
        queries_dir = sys.argv[1]

    print("Tree-sitter Query File Verification")
    print("="*40)
    print(f"Scanning directory: {queries_dir}")

    verifier = QueryVerifier(queries_dir)
    results = verifier.verify_all_files()
    verifier.print_results(results)

    # Exit with error code if there are errors
    if results['files_with_errors'] > 0:
        sys.exit(1)
    else:
        print(f"\n✅ All query files verified successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
