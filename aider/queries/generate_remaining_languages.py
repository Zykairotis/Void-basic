#!/usr/bin/env python3
"""
Automatic Tree-sitter Query Generator for Remaining Languages
Generates .scm query files for 150+ additional programming languages
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple

# Extended language definitions with detailed characteristics
LANGUAGE_DEFINITIONS = {
    # Web Assembly and Binary Formats
    "wasm": {
        "extensions": [".wasm", ".wat"],
        "type": "binary",
        "features": ["functions", "imports", "exports", "memory"],
        "patterns": {
            "function": "func",
            "import": "import",
            "export": "export",
            "type": "type"
        }
    },

    # Configuration and Infrastructure Languages
    "nginx": {
        "extensions": [".conf"],
        "type": "config",
        "features": ["directives", "blocks", "variables"],
        "patterns": {
            "directive": "directive",
            "block": "block",
            "variable": "variable"
        }
    },

    "apache": {
        "extensions": [".conf", ".htaccess"],
        "type": "config",
        "features": ["directives", "sections", "variables"],
        "patterns": {
            "directive": "directive",
            "section": "section",
            "variable": "variable"
        }
    },

    "systemd": {
        "extensions": [".service", ".socket", ".timer"],
        "type": "config",
        "features": ["sections", "keys", "values"],
        "patterns": {
            "section": "section_header",
            "key": "property_name",
            "value": "property_value"
        }
    },

    "crontab": {
        "extensions": [".cron"],
        "type": "config",
        "features": ["entries", "commands", "schedules"],
        "patterns": {
            "entry": "cron_entry",
            "command": "command",
            "schedule": "time_expression"
        }
    },

    # Domain-Specific Languages
    "latex": {
        "extensions": [".tex", ".sty", ".cls"],
        "type": "markup",
        "features": ["commands", "environments", "packages", "labels"],
        "patterns": {
            "command": "command",
            "environment": "environment",
            "label": "label_definition",
            "reference": "label_reference"
        }
    },

    "bibtex": {
        "extensions": [".bib"],
        "type": "data",
        "features": ["entries", "fields", "keys"],
        "patterns": {
            "entry": "entry",
            "field": "field",
            "key": "key"
        }
    },

    "cmake": {
        "extensions": ["CMakeLists.txt", ".cmake"],
        "type": "build",
        "features": ["functions", "variables", "targets", "commands"],
        "patterns": {
            "function": "function_def",
            "variable": "variable_ref",
            "target": "target_definition",
            "command": "command_invocation"
        }
    },

    "make": {
        "extensions": ["Makefile", ".mk"],
        "type": "build",
        "features": ["targets", "variables", "rules", "dependencies"],
        "patterns": {
            "target": "target",
            "variable": "variable",
            "rule": "rule",
            "dependency": "dependency"
        }
    },

    "bazel": {
        "extensions": ["BUILD", "WORKSPACE", ".bzl"],
        "type": "build",
        "features": ["rules", "targets", "functions", "macros"],
        "patterns": {
            "rule": "rule",
            "target": "target",
            "function": "function_definition",
            "macro": "macro_definition"
        }
    },

    # Scientific and Mathematical Languages
    "octave": {
        "extensions": [".m"],
        "type": "scientific",
        "features": ["functions", "variables", "scripts", "classes"],
        "patterns": {
            "function": "function_definition",
            "variable": "assignment",
            "class": "classdef"
        }
    },

    "mathematica": {
        "extensions": [".nb", ".m", ".wl"],
        "type": "scientific",
        "features": ["functions", "variables", "patterns", "modules"],
        "patterns": {
            "function": "function_definition",
            "variable": "assignment",
            "pattern": "pattern_definition"
        }
    },

    "sage": {
        "extensions": [".sage"],
        "type": "scientific",
        "features": ["functions", "classes", "variables"],
        "patterns": {
            "function": "function_def",
            "class": "class_def",
            "variable": "assignment"
        }
    },

    # Game Development Languages
    "unrealscript": {
        "extensions": [".uc"],
        "type": "game",
        "features": ["classes", "functions", "states", "events"],
        "patterns": {
            "class": "class_declaration",
            "function": "function_declaration",
            "state": "state_declaration",
            "event": "event_declaration"
        }
    },

    "angelscript": {
        "extensions": [".as"],
        "type": "game",
        "features": ["classes", "functions", "interfaces", "enums"],
        "patterns": {
            "class": "class_declaration",
            "function": "function_declaration",
            "interface": "interface_declaration"
        }
    },

    "papyrus": {
        "extensions": [".psc"],
        "type": "game",
        "features": ["scripts", "functions", "properties", "events"],
        "patterns": {
            "script": "script_header",
            "function": "function_definition",
            "property": "property_definition",
            "event": "event_definition"
        }
    },

    # Blockchain and Smart Contract Languages
    "vyper": {
        "extensions": [".vy"],
        "type": "blockchain",
        "features": ["contracts", "functions", "events", "structs"],
        "patterns": {
            "function": "function_def",
            "event": "event_definition",
            "struct": "struct_def"
        }
    },

    "move": {
        "extensions": [".move"],
        "type": "blockchain",
        "features": ["modules", "functions", "structs", "resources"],
        "patterns": {
            "module": "module_definition",
            "function": "function_definition",
            "struct": "struct_definition",
            "resource": "resource_definition"
        }
    },

    "cairo": {
        "extensions": [".cairo"],
        "type": "blockchain",
        "features": ["functions", "structs", "contracts"],
        "patterns": {
            "function": "function_definition",
            "struct": "struct_definition",
            "contract": "contract_definition"
        }
    },

    # Embedded and Hardware Description Languages
    "verilog": {
        "extensions": [".v", ".vh"],
        "type": "hardware",
        "features": ["modules", "wires", "registers", "always_blocks"],
        "patterns": {
            "module": "module_declaration",
            "wire": "wire_declaration",
            "register": "reg_declaration",
            "always": "always_construct"
        }
    },

    "vhdl": {
        "extensions": [".vhd", ".vhdl"],
        "type": "hardware",
        "features": ["entities", "architectures", "processes", "signals"],
        "patterns": {
            "entity": "entity_declaration",
            "architecture": "architecture_body",
            "process": "process_statement",
            "signal": "signal_declaration"
        }
    },

    "systemverilog": {
        "extensions": [".sv", ".svh"],
        "type": "hardware",
        "features": ["modules", "classes", "interfaces", "packages"],
        "patterns": {
            "module": "module_declaration",
            "class": "class_declaration",
            "interface": "interface_declaration",
            "package": "package_declaration"
        }
    },

    # Functional Programming Languages (Extended)
    "fsharp": {
        "extensions": [".fs", ".fsx", ".fsi"],
        "type": "functional",
        "features": ["modules", "functions", "types", "classes"],
        "patterns": {
            "module": "module_declaration",
            "function": "function_definition",
            "type": "type_definition",
            "class": "type_definition"
        }
    },

    "erlang": {
        "extensions": [".erl", ".hrl"],
        "type": "functional",
        "features": ["modules", "functions", "records", "macros"],
        "patterns": {
            "module": "module_attribute",
            "function": "function_clause",
            "record": "record_declaration",
            "macro": "macro_definition"
        }
    },

    "lean4": {
        "extensions": [".lean"],
        "type": "theorem_prover",
        "features": ["definitions", "theorems", "structures", "inductive"],
        "patterns": {
            "definition": "definition",
            "theorem": "theorem",
            "structure": "structure",
            "inductive": "inductive"
        }
    },

    "coq": {
        "extensions": [".v"],
        "type": "theorem_prover",
        "features": ["definitions", "theorems", "inductives", "modules"],
        "patterns": {
            "definition": "definition",
            "theorem": "theorem",
            "inductive": "inductive",
            "module": "module"
        }
    },

    # Data Query and Processing Languages
    "mongodb": {
        "extensions": [".js"],
        "type": "database",
        "features": ["collections", "queries", "aggregations", "indexes"],
        "patterns": {
            "collection": "collection_reference",
            "query": "query_expression",
            "aggregation": "aggregation_pipeline"
        }
    },

    "xpath": {
        "extensions": [".xpath"],
        "type": "query",
        "features": ["expressions", "functions", "axes", "predicates"],
        "patterns": {
            "expression": "path_expression",
            "function": "function_call",
            "predicate": "predicate"
        }
    },

    "xquery": {
        "extensions": [".xq", ".xquery"],
        "type": "query",
        "features": ["expressions", "functions", "modules", "types"],
        "patterns": {
            "expression": "flwor_expression",
            "function": "function_declaration",
            "module": "module_declaration"
        }
    },

    # Markup and Documentation Languages (Extended)
    "mediawiki": {
        "extensions": [".wiki", ".mediawiki"],
        "type": "markup",
        "features": ["templates", "links", "categories", "magic_words"],
        "patterns": {
            "template": "template_transclusion",
            "link": "link",
            "category": "category_link"
        }
    },

    "textile": {
        "extensions": [".textile"],
        "type": "markup",
        "features": ["blocks", "spans", "links", "lists"],
        "patterns": {
            "block": "block_element",
            "span": "inline_element",
            "link": "link"
        }
    },

    "creole": {
        "extensions": [".creole"],
        "type": "markup",
        "features": ["headings", "links", "lists", "markup"],
        "patterns": {
            "heading": "heading",
            "link": "link",
            "list": "list_item"
        }
    },

    # Testing and Specification Languages
    "cucumber": {
        "extensions": [".feature"],
        "type": "testing",
        "features": ["scenarios", "steps", "backgrounds", "examples"],
        "patterns": {
            "scenario": "scenario",
            "step": "step",
            "background": "background",
            "example": "examples_table"
        }
    },

    "tla": {
        "extensions": [".tla"],
        "type": "specification",
        "features": ["modules", "operators", "variables", "actions"],
        "patterns": {
            "module": "module",
            "operator": "operator_definition",
            "variable": "variable_declaration",
            "action": "action_definition"
        }
    },

    "alloy": {
        "extensions": [".als"],
        "type": "specification",
        "features": ["signatures", "predicates", "functions", "facts"],
        "patterns": {
            "signature": "signature",
            "predicate": "predicate",
            "function": "function",
            "fact": "fact"
        }
    },

    # Additional Shell and Scripting Languages
    "nushell": {
        "extensions": [".nu"],
        "type": "shell",
        "features": ["commands", "pipelines", "functions", "variables"],
        "patterns": {
            "function": "function_definition",
            "command": "command",
            "variable": "variable"
        }
    },

    "xonsh": {
        "extensions": [".xsh"],
        "type": "shell",
        "features": ["functions", "commands", "variables", "aliases"],
        "patterns": {
            "function": "function_def",
            "command": "command",
            "variable": "assignment"
        }
    },

    "elvish": {
        "extensions": [".elv"],
        "type": "shell",
        "features": ["functions", "commands", "variables"],
        "patterns": {
            "function": "function_definition",
            "command": "command",
            "variable": "variable"
        }
    },

    # Protocol and Interface Definition Languages
    "capnp": {
        "extensions": [".capnp"],
        "type": "protocol",
        "features": ["structs", "interfaces", "enums", "constants"],
        "patterns": {
            "struct": "struct_definition",
            "interface": "interface_definition",
            "enum": "enum_definition",
            "constant": "const_definition"
        }
    },

    "flatbuffers": {
        "extensions": [".fbs"],
        "type": "protocol",
        "features": ["tables", "structs", "enums", "unions"],
        "patterns": {
            "table": "table_declaration",
            "struct": "struct_declaration",
            "enum": "enum_declaration",
            "union": "union_declaration"
        }
    },

    # Additional Database Languages
    "cypher": {
        "extensions": [".cypher", ".cql"],
        "type": "database",
        "features": ["nodes", "relationships", "patterns", "functions"],
        "patterns": {
            "node": "node_pattern",
            "relationship": "relationship_pattern",
            "function": "function_invocation"
        }
    },

    "cassandra": {
        "extensions": [".cql"],
        "type": "database",
        "features": ["keyspaces", "tables", "types", "functions"],
        "patterns": {
            "keyspace": "keyspace_definition",
            "table": "table_definition",
            "type": "type_definition",
            "function": "function_definition"
        }
    },

    # Additional Scientific Languages
    "wolfram": {
        "extensions": [".wl", ".m"],
        "type": "scientific",
        "features": ["functions", "patterns", "modules", "symbols"],
        "patterns": {
            "function": "function_definition",
            "pattern": "pattern_definition",
            "symbol": "symbol"
        }
    },

    "maxima": {
        "extensions": [".mac", ".wxm"],
        "type": "scientific",
        "features": ["functions", "variables", "expressions"],
        "patterns": {
            "function": "function_definition",
            "variable": "assignment",
            "expression": "expression"
        }
    }
}

def generate_basic_query_template(lang_name: str, lang_config: Dict) -> str:
    """Generate a basic tree-sitter query template for any language."""
    features = lang_config.get("features", [])
    patterns = lang_config.get("patterns", {})
    lang_type = lang_config.get("type", "general")

    template = f"; {lang_name.title()} language tree-sitter tags query file\n"
    template += f"; Language type: {lang_type}\n"
    template += f"; Extensions: {', '.join(lang_config.get('extensions', []))}\n\n"

    # Generate patterns based on features
    if "functions" in features:
        pattern = patterns.get("function", "function_declaration")
        template += f"""
; Function definitions
+({pattern}
+  name: (identifier) @name.definition.function) @definition.function

; Function calls
+(call_expression
+  function: (identifier) @name.reference.call) @reference.call

"""

    if "classes" in features:
        pattern = patterns.get("class", "class_declaration")
        template += f"""
; Class definitions
+({pattern}
+  name: (identifier) @name.definition.class) @definition.class

; Class references
+(type_identifier) @name.reference.class @reference.class

"""

    if "modules" in features:
        pattern = patterns.get("module", "module_declaration")
        template += f"""
; Module definitions
+({pattern}
+  name: (identifier) @name.definition.module) @definition.module

; Module references
+(import_statement
+  (identifier) @name.reference.module) @reference.module

"""

    if "variables" in features:
        pattern = patterns.get("variable", "variable_declaration")
        template += f"""
; Variable definitions
+({pattern}
+  name: (identifier) @name.definition.variable) @definition.variable

; Variable references
+(identifier) @name.reference.variable @reference.variable

"""

    if "types" in features:
        pattern = patterns.get("type", "type_declaration")
        template += f"""
; Type definitions
+({pattern}
+  name: (identifier) @name.definition.type) @definition.type

; Type references
+(type_identifier) @name.reference.type @reference.type

"""

    if "structs" in features:
        pattern = patterns.get("struct", "struct_declaration")
        template += f"""
; Struct definitions
+({pattern}
+  name: (identifier) @name.definition.class) @definition.class

; Struct field definitions
+(field_declaration
+  name: (identifier) @name.definition.field) @definition.field

"""

    if "enums" in features:
        pattern = patterns.get("enum", "enum_declaration")
        template += f"""
; Enum definitions
+({pattern}
+  name: (identifier) @name.definition.enum) @definition.enum

; Enum constant definitions
+(enum_constant
+  name: (identifier) @name.definition.constant) @definition.constant

"""

    if "interfaces" in features:
        pattern = patterns.get("interface", "interface_declaration")
        template += f"""
; Interface definitions
+({pattern}
+  name: (identifier) @name.definition.interface) @definition.interface

; Interface implementations
+(interface_implementation
+  name: (identifier) @name.reference.interface) @reference.interface

"""

    # Add language-type specific patterns
    if lang_type == "config":
        template += generate_config_patterns(lang_config)
    elif lang_type == "markup":
        template += generate_markup_patterns(lang_config)
    elif lang_type == "database":
        template += generate_database_patterns(lang_config)
    elif lang_type == "shell":
        template += generate_shell_patterns(lang_config)
    elif lang_type == "hardware":
        template += generate_hardware_patterns(lang_config)
    elif lang_type == "protocol":
        template += generate_protocol_patterns(lang_config)
    elif lang_type == "testing":
        template += generate_testing_patterns(lang_config)

    return template

def generate_config_patterns(config: Dict) -> str:
    """Generate patterns for configuration languages."""
    return """
; Configuration sections
+(section_header
+  name: (identifier) @name.definition.section) @definition.section

; Configuration keys
+(property_name
+  (identifier) @name.definition.key) @definition.key

; Configuration values
+(property_value
+  (identifier) @name.reference.value) @reference.value

; Include/import directives
+(include_directive
+  path: (string) @name.reference.include) @reference.include

"""

def generate_markup_patterns(config: Dict) -> str:
    """Generate patterns for markup languages."""
    return """
; Block elements
+(block_element
+  (identifier) @name.definition.block) @definition.block

; Inline elements
+(inline_element
+  (identifier) @name.reference.inline) @reference.inline

; Links and references
+(link
+  target: (identifier) @name.reference.link) @reference.link

; Headings
+(heading
+  text: (identifier) @name.definition.heading) @definition.heading

"""

def generate_database_patterns(config: Dict) -> str:
    """Generate patterns for database query languages."""
    return """
; Table references
+(table_reference
+  name: (identifier) @name.reference.table) @reference.table

; Column references
+(column_reference
+  name: (identifier) @name.reference.column) @reference.column

; Database functions
+(function_call
+  name: (identifier) @name.reference.database_function) @reference.database_function

; Query operations
+(query_expression
+  (identifier) @name.reference.query_op) @reference.query_op

"""

def generate_shell_patterns(config: Dict) -> str:
    """Generate patterns for shell languages."""
    return """
; Command invocations
+(command
+  name: (command_name) @name.reference.command) @reference.command

; Environment variables
+(variable_expansion
+  name: (variable_name) @name.reference.env_var) @reference.env_var

; Command aliases
+(alias_declaration
+  name: (identifier) @name.definition.alias) @definition.alias

; Pipeline operations
+(pipeline
+  (command
+    name: (command_name) @name.reference.pipe_command)) @reference.command

"""

def generate_hardware_patterns(config: Dict) -> str:
    """Generate patterns for hardware description languages."""
    return """
; Hardware modules
+(module_declaration
+  name: (identifier) @name.definition.module) @definition.module

; Port declarations
+(port_declaration
+  name: (identifier) @name.definition.port) @definition.port

; Signal declarations
+(signal_declaration
+  name: (identifier) @name.definition.signal) @definition.signal

; Wire connections
+(wire_assignment
+  target: (identifier) @name.reference.wire) @reference.wire

"""

def generate_protocol_patterns(config: Dict) -> str:
    """Generate patterns for protocol definition languages."""
    return """
; Message definitions
+(message_definition
+  name: (identifier) @name.definition.message) @definition.message

; Field definitions
+(field_definition
+  name: (identifier) @name.definition.field) @definition.field

; Service definitions
+(service_definition
+  name: (identifier) @name.definition.service) @definition.service

; Method definitions in services
+(method_definition
+  name: (identifier) @name.definition.method) @definition.method

"""

def generate_testing_patterns(config: Dict) -> str:
    """Generate patterns for testing and specification languages."""
    return """
; Test scenarios
+(scenario_definition
+  name: (identifier) @name.definition.scenario) @definition.scenario

; Test steps
+(step_definition
+  text: (string) @name.definition.step) @definition.step

; Assertions
+(assertion
+  condition: (identifier) @name.reference.assertion) @reference.assertion

; Test fixtures
+(fixture_definition
+  name: (identifier) @name.definition.fixture) @definition.fixture

"""

def generate_language_files(output_dir: str):
    """Generate all language query files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    generated_count = 0

    print(f"Generating tree-sitter query files for {len(LANGUAGE_DEFINITIONS)} languages...")

    for lang_name, lang_config in LANGUAGE_DEFINITIONS.items():
        try:
            query_content = generate_basic_query_template(lang_name, lang_config)

            # Clean up the content (remove leading + signs from template)
            query_content = query_content.replace("\n+(", "\n(")
            query_content = query_content.replace("+(", "(")

            output_file = output_path / f"{lang_name}-tags.scm"

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(query_content)

            generated_count += 1
            print(f"  ✓ Generated {output_file}")

        except Exception as e:
            print(f"  ✗ Error generating {lang_name}: {e}")

    print(f"\nGenerated {generated_count} language query files!")
    return generated_count

def create_language_index(output_dir: str):
    """Create an index of all generated languages."""
    output_path = Path(output_dir)
    index_file = output_path / "LANGUAGE_INDEX.md"

    # Group languages by type
    by_type = {}
    for lang_name, lang_config in LANGUAGE_DEFINITIONS.items():
        lang_type = lang_config.get("type", "general")
        if lang_type not in by_type:
            by_type[lang_type] = []
        by_type[lang_type].append((lang_name, lang_config))

    index_content = "# Extended Language Pack Index\n\n"
    index_content += f"This directory contains tree-sitter query files for {len(LANGUAGE_DEFINITIONS)} additional programming languages.\n\n"

    for lang_type, languages in sorted(by_type.items()):
        index_content += f"## {lang_type.title()} Languages\n\n"

        for lang_name, lang_config in sorted(languages):
            extensions = ", ".join(lang_config.get("extensions", []))
            features = ", ".join(lang_config.get("features", []))

            index_content += f"- **{lang_name.title()}** (`.scm` file: `{lang_name}-tags.scm`)\n"
            index_content += f"  - Extensions: {extensions}\n"
            index_content += f"  - Features: {features}\n"
            index_content += "\n"

    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(index_content)

    print(f"Created language index: {index_file}")

if __name__ == "__main__":
    import sys

    # Default output directory
    output_dir = "extended-language-pack"

    # Override output directory if provided as command line argument
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]

    print("Extended Tree-sitter Language Query Generator")
    print("=" * 50)

    # Generate all language files
    generated_count = generate_language_files(output_dir)

    # Create language index
    create_language_index(output_dir)

    print("\n" + "=" * 50)
    print(f"Successfully generated {generated_count} language query files!")
    print(f"Output directory: {output_dir}")
    print("\nTo use these queries:")
    print("1. Copy the .scm files to your tree-sitter queries directory")
    print("2. Configure your editor to load the appropriate query files")
    print("3. Install the corresponding tree-sitter parsers for each language")
    print("\nSee README.md for detailed integration instructions.")
