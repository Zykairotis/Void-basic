#!/usr/bin/env python3
"""
Comprehensive Tree-sitter Query Generator for 200+ Programming Languages
Generates .scm query files for code navigation, highlighting, and analysis.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Language definitions with their characteristics
LANGUAGES = {
    # Systems Programming
    "zig": {
        "type": "systems",
        "extensions": [".zig"],
        "has_classes": False,
        "has_functions": True,
        "has_structs": True,
        "has_enums": True,
        "function_call_pattern": "call_expression",
        "function_def_pattern": "function_declaration"
    },
    "v": {
        "type": "systems",
        "extensions": [".v"],
        "has_classes": False,
        "has_functions": True,
        "has_structs": True,
        "has_enums": True,
        "function_call_pattern": "call_expression",
        "function_def_pattern": "function_declaration"
    },
    "odin": {
        "type": "systems",
        "extensions": [".odin"],
        "has_classes": False,
        "has_functions": True,
        "has_structs": True,
        "has_enums": True,
        "function_call_pattern": "call_expression",
        "function_def_pattern": "procedure_declaration"
    },
    "carbon": {
        "type": "systems",
        "extensions": [".carbon"],
        "has_classes": True,
        "has_functions": True,
        "has_structs": True,
        "has_enums": True,
        "function_call_pattern": "call_expression",
        "function_def_pattern": "function_declaration"
    },

    # Functional Languages
    "haskell": {
        "type": "functional",
        "extensions": [".hs", ".lhs"],
        "has_classes": False,
        "has_functions": True,
        "has_types": True,
        "has_modules": True,
        "function_call_pattern": "application_expression",
        "function_def_pattern": "function_declaration"
    },
    "ocaml": {
        "type": "functional",
        "extensions": [".ml", ".mli"],
        "has_classes": True,
        "has_functions": True,
        "has_modules": True,
        "function_call_pattern": "application_expression",
        "function_def_pattern": "value_definition"
    },
    "fsharp": {
        "type": "functional",
        "extensions": [".fs", ".fsx", ".fsi"],
        "has_classes": True,
        "has_functions": True,
        "has_modules": True,
        "function_call_pattern": "application_expression",
        "function_def_pattern": "function_declaration"
    },
    "purescript": {
        "type": "functional",
        "extensions": [".purs"],
        "has_classes": False,
        "has_functions": True,
        "has_modules": True,
        "function_call_pattern": "application_expression",
        "function_def_pattern": "value_declaration"
    },
    "elm": {
        "type": "functional",
        "extensions": [".elm"],
        "has_classes": False,
        "has_functions": True,
        "has_modules": True,
        "function_call_pattern": "function_call_expr",
        "function_def_pattern": "value_declaration"
    },
    "idris": {
        "type": "functional",
        "extensions": [".idr", ".lidr"],
        "has_classes": False,
        "has_functions": True,
        "has_types": True,
        "function_call_pattern": "application_expression",
        "function_def_pattern": "function_declaration"
    },
    "lean": {
        "type": "functional",
        "extensions": [".lean"],
        "has_classes": False,
        "has_functions": True,
        "has_types": True,
        "function_call_pattern": "application_expression",
        "function_def_pattern": "definition"
    },

    # Scripting Languages
    "lua": {
        "type": "scripting",
        "extensions": [".lua"],
        "has_classes": False,
        "has_functions": True,
        "has_tables": True,
        "function_call_pattern": "function_call",
        "function_def_pattern": "function_declaration"
    },
    "perl": {
        "type": "scripting",
        "extensions": [".pl", ".pm", ".perl"],
        "has_classes": True,
        "has_functions": True,
        "has_packages": True,
        "function_call_pattern": "function_call",
        "function_def_pattern": "subroutine_declaration"
    },
    "tcl": {
        "type": "scripting",
        "extensions": [".tcl", ".tk"],
        "has_classes": False,
        "has_functions": True,
        "has_procedures": True,
        "function_call_pattern": "command",
        "function_def_pattern": "proc_declaration"
    },

    # Web Technologies
    "html": {
        "type": "markup",
        "extensions": [".html", ".htm"],
        "has_elements": True,
        "has_attributes": True,
        "element_pattern": "element",
        "attribute_pattern": "attribute"
    },
    "css": {
        "type": "stylesheet",
        "extensions": [".css"],
        "has_selectors": True,
        "has_properties": True,
        "has_rules": True,
        "selector_pattern": "selector",
        "property_pattern": "property_name"
    },
    "scss": {
        "type": "stylesheet",
        "extensions": [".scss"],
        "has_selectors": True,
        "has_properties": True,
        "has_mixins": True,
        "has_variables": True,
        "selector_pattern": "selector",
        "property_pattern": "property_name"
    },
    "less": {
        "type": "stylesheet",
        "extensions": [".less"],
        "has_selectors": True,
        "has_properties": True,
        "has_mixins": True,
        "has_variables": True,
        "selector_pattern": "selector",
        "property_pattern": "property_name"
    },
    "stylus": {
        "type": "stylesheet",
        "extensions": [".styl"],
        "has_selectors": True,
        "has_properties": True,
        "has_mixins": True,
        "has_variables": True,
        "selector_pattern": "selector",
        "property_pattern": "property_name"
    },
    "vue": {
        "type": "framework",
        "extensions": [".vue"],
        "has_components": True,
        "has_templates": True,
        "has_scripts": True,
        "component_pattern": "component",
        "directive_pattern": "directive"
    },
    "svelte": {
        "type": "framework",
        "extensions": [".svelte"],
        "has_components": True,
        "has_templates": True,
        "has_scripts": True,
        "component_pattern": "component",
        "directive_pattern": "directive"
    },

    # Database Languages
    "sql": {
        "type": "database",
        "extensions": [".sql"],
        "has_tables": True,
        "has_procedures": True,
        "has_functions": True,
        "table_pattern": "table_reference",
        "function_pattern": "function_call"
    },
    "postgresql": {
        "type": "database",
        "extensions": [".sql", ".psql"],
        "has_tables": True,
        "has_procedures": True,
        "has_functions": True,
        "table_pattern": "table_reference",
        "function_pattern": "function_call"
    },
    "mysql": {
        "type": "database",
        "extensions": [".sql"],
        "has_tables": True,
        "has_procedures": True,
        "has_functions": True,
        "table_pattern": "table_reference",
        "function_pattern": "function_call"
    },

    # Configuration Languages
    "yaml": {
        "type": "config",
        "extensions": [".yaml", ".yml"],
        "has_keys": True,
        "has_values": True,
        "key_pattern": "block_mapping_pair",
        "value_pattern": "plain_scalar"
    },
    "toml": {
        "type": "config",
        "extensions": [".toml"],
        "has_keys": True,
        "has_sections": True,
        "key_pattern": "pair",
        "section_pattern": "table"
    },
    "ini": {
        "type": "config",
        "extensions": [".ini", ".cfg", ".conf"],
        "has_keys": True,
        "has_sections": True,
        "key_pattern": "setting",
        "section_pattern": "section"
    },
    "json": {
        "type": "data",
        "extensions": [".json"],
        "has_objects": True,
        "has_arrays": True,
        "object_pattern": "object",
        "array_pattern": "array"
    },
    "json5": {
        "type": "data",
        "extensions": [".json5"],
        "has_objects": True,
        "has_arrays": True,
        "object_pattern": "object",
        "array_pattern": "array"
    },
    "jsonc": {
        "type": "data",
        "extensions": [".jsonc"],
        "has_objects": True,
        "has_arrays": True,
        "object_pattern": "object",
        "array_pattern": "array"
    },

    # Shell Languages
    "bash": {
        "type": "shell",
        "extensions": [".sh", ".bash"],
        "has_functions": True,
        "has_variables": True,
        "has_commands": True,
        "function_pattern": "function_definition",
        "command_pattern": "command"
    },
    "zsh": {
        "type": "shell",
        "extensions": [".zsh"],
        "has_functions": True,
        "has_variables": True,
        "has_commands": True,
        "function_pattern": "function_definition",
        "command_pattern": "command"
    },
    "fish": {
        "type": "shell",
        "extensions": [".fish"],
        "has_functions": True,
        "has_variables": True,
        "has_commands": True,
        "function_pattern": "function_definition",
        "command_pattern": "command"
    },
    "powershell": {
        "type": "shell",
        "extensions": [".ps1", ".psm1", ".psd1"],
        "has_functions": True,
        "has_classes": True,
        "has_cmdlets": True,
        "function_pattern": "function_statement",
        "class_pattern": "class_statement"
    },

    # JVM Languages
    "scala": {
        "type": "jvm",
        "extensions": [".scala", ".sc"],
        "has_classes": True,
        "has_functions": True,
        "has_objects": True,
        "has_traits": True,
        "class_pattern": "class_definition",
        "function_pattern": "function_definition"
    },
    "kotlin": {
        "type": "jvm",
        "extensions": [".kt", ".kts"],
        "has_classes": True,
        "has_functions": True,
        "has_interfaces": True,
        "class_pattern": "class_declaration",
        "function_pattern": "function_declaration"
    },
    "groovy": {
        "type": "jvm",
        "extensions": [".groovy", ".gvy", ".gy", ".gsh"],
        "has_classes": True,
        "has_functions": True,
        "has_closures": True,
        "class_pattern": "class_declaration",
        "function_pattern": "method_declaration"
    },
    "clojure": {
        "type": "jvm",
        "extensions": [".clj", ".cljs", ".cljc"],
        "has_functions": True,
        "has_macros": True,
        "has_namespaces": True,
        "function_pattern": "list_lit",
        "namespace_pattern": "list_lit"
    },

    # Mobile Development
    "swift": {
        "type": "mobile",
        "extensions": [".swift"],
        "has_classes": True,
        "has_functions": True,
        "has_protocols": True,
        "has_structs": True,
        "class_pattern": "class_declaration",
        "function_pattern": "function_declaration"
    },
    "objective_c": {
        "type": "mobile",
        "extensions": [".m", ".mm"],
        "has_classes": True,
        "has_functions": True,
        "has_protocols": True,
        "class_pattern": "class_interface",
        "function_pattern": "method_declaration"
    },
    "dart": {
        "type": "mobile",
        "extensions": [".dart"],
        "has_classes": True,
        "has_functions": True,
        "has_mixins": True,
        "class_pattern": "class_definition",
        "function_pattern": "function_signature"
    },

    # Game Development
    "gdscript": {
        "type": "game",
        "extensions": [".gd"],
        "has_classes": True,
        "has_functions": True,
        "has_signals": True,
        "class_pattern": "class_declaration",
        "function_pattern": "function_definition"
    },
    "hlsl": {
        "type": "shader",
        "extensions": [".hlsl", ".fx"],
        "has_functions": True,
        "has_structs": True,
        "has_shaders": True,
        "function_pattern": "function_declaration",
        "struct_pattern": "struct_declaration"
    },
    "glsl": {
        "type": "shader",
        "extensions": [".glsl", ".vert", ".frag", ".geom", ".tesc", ".tese", ".comp"],
        "has_functions": True,
        "has_structs": True,
        "has_shaders": True,
        "function_pattern": "function_declaration",
        "struct_pattern": "struct_declaration"
    },
    "wgsl": {
        "type": "shader",
        "extensions": [".wgsl"],
        "has_functions": True,
        "has_structs": True,
        "has_shaders": True,
        "function_pattern": "function_declaration",
        "struct_pattern": "struct_declaration"
    },

    # Data Science
    "r": {
        "type": "statistics",
        "extensions": [".r", ".R"],
        "has_functions": True,
        "has_variables": True,
        "has_packages": True,
        "function_pattern": "function_definition",
        "call_pattern": "call"
    },
    "julia": {
        "type": "scientific",
        "extensions": [".jl"],
        "has_functions": True,
        "has_modules": True,
        "has_types": True,
        "has_structs": True,
        "function_pattern": "function_definition",
        "module_pattern": "module_definition"
    },
    "matlab": {
        "type": "scientific",
        "extensions": [".m"],
        "has_functions": True,
        "has_classes": True,
        "has_scripts": True,
        "function_pattern": "function_definition",
        "class_pattern": "class_definition"
    },

    # Documentation
    "markdown": {
        "type": "documentation",
        "extensions": [".md", ".markdown"],
        "has_headings": True,
        "has_links": True,
        "has_code": True,
        "heading_pattern": "atx_heading",
        "link_pattern": "link"
    },
    "rst": {
        "type": "documentation",
        "extensions": [".rst", ".rest"],
        "has_headings": True,
        "has_links": True,
        "has_code": True,
        "heading_pattern": "title",
        "link_pattern": "reference"
    },
    "asciidoc": {
        "type": "documentation",
        "extensions": [".adoc", ".asciidoc"],
        "has_headings": True,
        "has_links": True,
        "has_code": True,
        "heading_pattern": "section_title",
        "link_pattern": "link"
    },
    "org": {
        "type": "documentation",
        "extensions": [".org"],
        "has_headings": True,
        "has_links": True,
        "has_code": True,
        "heading_pattern": "headline",
        "link_pattern": "link"
    },

    # Emerging Languages
    "zig": {
        "type": "systems",
        "extensions": [".zig"],
        "has_functions": True,
        "has_structs": True,
        "has_enums": True,
        "function_pattern": "function_declaration",
        "struct_pattern": "struct_declaration"
    },
    "nim": {
        "type": "systems",
        "extensions": [".nim"],
        "has_functions": True,
        "has_types": True,
        "has_modules": True,
        "function_pattern": "proc_declaration",
        "type_pattern": "type_declaration"
    },
    "crystal": {
        "type": "systems",
        "extensions": [".cr"],
        "has_classes": True,
        "has_functions": True,
        "has_modules": True,
        "class_pattern": "class_definition",
        "function_pattern": "def"
    },
    "reason": {
        "type": "functional",
        "extensions": [".re", ".rei"],
        "has_functions": True,
        "has_modules": True,
        "has_types": True,
        "function_pattern": "value_definition",
        "module_pattern": "module_declaration"
    },

    # Query Languages
    "graphql": {
        "type": "query",
        "extensions": [".graphql", ".gql"],
        "has_types": True,
        "has_queries": True,
        "has_mutations": True,
        "type_pattern": "type_definition",
        "query_pattern": "operation_definition"
    },
    "sparql": {
        "type": "query",
        "extensions": [".sparql", ".rq"],
        "has_queries": True,
        "has_prefixes": True,
        "query_pattern": "query",
        "prefix_pattern": "prefix_declaration"
    },

    # Infrastructure as Code
    "terraform": {
        "type": "iac",
        "extensions": [".tf", ".tfvars"],
        "has_resources": True,
        "has_variables": True,
        "has_modules": True,
        "resource_pattern": "block",
        "variable_pattern": "attribute"
    },
    "ansible": {
        "type": "iac",
        "extensions": [".yml", ".yaml"],
        "has_tasks": True,
        "has_roles": True,
        "has_variables": True,
        "task_pattern": "block_mapping",
        "variable_pattern": "flow_mapping"
    },
    "kubernetes": {
        "type": "iac",
        "extensions": [".yaml", ".yml"],
        "has_resources": True,
        "has_metadata": True,
        "has_specs": True,
        "resource_pattern": "block_mapping",
        "metadata_pattern": "block_mapping"
    },

    # Protocol Definition
    "protobuf": {
        "type": "protocol",
        "extensions": [".proto"],
        "has_messages": True,
        "has_services": True,
        "has_enums": True,
        "message_pattern": "message",
        "service_pattern": "service"
    },
    "thrift": {
        "type": "protocol",
        "extensions": [".thrift"],
        "has_structs": True,
        "has_services": True,
        "has_enums": True,
        "struct_pattern": "struct",
        "service_pattern": "service"
    },
    "avro": {
        "type": "protocol",
        "extensions": [".avsc", ".avpr"],
        "has_schemas": True,
        "has_records": True,
        "has_enums": True,
        "record_pattern": "record_declaration",
        "enum_pattern": "enum_declaration"
    },
}

# Template generators for different language types
class QueryTemplateGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_basic_template(self, lang_name: str, lang_config: Dict) -> str:
        """Generate basic query template for any language."""
        template = f"; {lang_name.title()} language tree-sitter tags query file\n\n"

        if lang_config.get("has_functions", False):
            template += self._add_function_patterns(lang_config)

        if lang_config.get("has_classes", False):
            template += self._add_class_patterns(lang_config)

        if lang_config.get("has_structs", False):
            template += self._add_struct_patterns(lang_config)

        if lang_config.get("has_enums", False):
            template += self._add_enum_patterns(lang_config)

        if lang_config.get("has_modules", False):
            template += self._add_module_patterns(lang_config)

        if lang_config.get("has_interfaces", False):
            template += self._add_interface_patterns(lang_config)

        if lang_config.get("has_variables", False):
            template += self._add_variable_patterns(lang_config)

        # Add call patterns
        template += self._add_call_patterns(lang_config)

        return template

    def _add_function_patterns(self, config: Dict) -> str:
        pattern = config.get("function_def_pattern", "function_declaration")
        call_pattern = config.get("function_call_pattern", "call_expression")

        return f"""
; Function definitions
({pattern}
  name: (identifier) @name.definition.function) @definition.function

; Function calls
({call_pattern}
  function: (identifier) @name.reference.call) @reference.call

"""

    def _add_class_patterns(self, config: Dict) -> str:
        pattern = config.get("class_pattern", "class_declaration")

        return f"""
; Class definitions
({pattern}
  name: (identifier) @name.definition.class) @definition.class

; Class references
(type_identifier) @name.reference.class @reference.class

"""

    def _add_struct_patterns(self, config: Dict) -> str:
        pattern = config.get("struct_pattern", "struct_declaration")

        return f"""
; Struct definitions
({pattern}
  name: (identifier) @name.definition.class) @definition.class

"""

    def _add_enum_patterns(self, config: Dict) -> str:
        pattern = config.get("enum_pattern", "enum_declaration")

        return f"""
; Enum definitions
({pattern}
  name: (identifier) @name.definition.enum) @definition.enum

; Enum constants
(enum_constant
  name: (identifier) @name.definition.constant) @definition.constant

"""

    def _add_module_patterns(self, config: Dict) -> str:
        pattern = config.get("module_pattern", "module_declaration")

        return f"""
; Module definitions
({pattern}
  name: (identifier) @name.definition.module) @definition.module

; Module references
(identifier) @name.reference.module @reference.module

"""

    def _add_interface_patterns(self, config: Dict) -> str:
        pattern = config.get("interface_pattern", "interface_declaration")

        return f"""
; Interface definitions
({pattern}
  name: (identifier) @name.definition.interface) @definition.interface

; Interface references
(type_identifier) @name.reference.interface @reference.interface

"""

    def _add_variable_patterns(self, config: Dict) -> str:
        pattern = config.get("variable_pattern", "variable_declaration")

        return f"""
; Variable definitions
({pattern}
  name: (identifier) @name.definition.variable) @definition.variable

; Variable references
(identifier) @name.reference.variable @reference.variable

"""

    def _add_call_patterns(self, config: Dict) -> str:
        if config.get("type") == "markup":
            return self._add_markup_patterns(config)
        elif config.get("type") == "stylesheet":
            return self._add_stylesheet_patterns(config)
        elif config.get("type") == "config":
            return self._add_config_patterns(config)
        elif config.get("type") == "database":
            return self._add_database_patterns(config)
        elif config.get("type") == "shell":
            return self._add_shell_patterns(config)
        else:
            return ""

    def _add_markup_patterns(self, config: Dict) -> str:
        return f"""
; HTML/XML Elements
(element
  (start_tag
    name: (tag_name) @name.definition.element)) @definition.element

; Attributes
(attribute
  name: (attribute_name) @name.definition.attribute) @definition.attribute

"""

    def _add_stylesheet_patterns(self, config: Dict) -> str:
        return f"""
; CSS Selectors
(rule_set
  (selectors
    (selector) @name.definition.selector)) @definition.selector

; CSS Properties
(declaration
  property: (property_name) @name.definition.property) @definition.property

; CSS Variables
(declaration
  property: (property_name) @name.definition.variable
  (#match? @name.definition.variable "^--")) @definition.variable

"""

    def _add_config_patterns(self, config: Dict) -> str:
        return f"""
; Configuration keys
(pair
  key: (identifier) @name.definition.key) @definition.key

; Configuration sections
(section
  name: (identifier) @name.definition.section) @definition.section

"""

    def _add_database_patterns(self, config: Dict) -> str:
        return f"""
; Table references
(table_reference
  name: (identifier) @name.reference.table) @reference.table

; Column references
(column_reference
  name: (identifier) @name.reference.column) @reference.column

; Stored procedures
(procedure_declaration
  name: (identifier) @name.definition.procedure) @definition.procedure

"""

    def _add_shell_patterns(self, config: Dict) -> str:
        return f"""
; Commands
(command
  name: (identifier) @name.reference.command) @reference.command

; Environment variables
(variable_expansion
  name: (identifier) @name.reference.variable) @reference.variable

"""

    def generate_specialized_template(self, lang_name: str, lang_config: Dict) -> str:
        """Generate specialized templates for specific language types."""
        lang_type = lang_config.get("type")

        if lang_type == "functional":
            return self._generate_functional_template(lang_name, lang_config)
        elif lang_type == "shell":
            return self._generate_shell_template(lang_name, lang_config)
        elif lang_type == "markup":
            return self._generate_markup_template(lang_name, lang_config)
        elif lang_type == "stylesheet":
            return self._generate_stylesheet_template(lang_name, lang_config)
        elif lang_type == "database":
            return self._generate_database_template(lang_name, lang_config)
        else:
            return self.generate_basic_template(lang_name, lang_config)

    def _generate_functional_template(self, lang_name: str, config: Dict) -> str:
        template = f"; {lang_name.title()} functional language tree-sitter tags query file\n\n"

        template += """
; Function definitions
(function_declaration
  name: (identifier) @name.definition.function) @definition.function

(value_declaration
  pattern: (identifier) @name.definition.function) @definition.function

; Type definitions
(type_declaration
  name: (type_identifier) @name.definition.type) @definition.type

(data_declaration
  name: (type_identifier) @name.definition.type) @definition.type

; Constructor definitions
(constructor_declaration
  name: (identifier) @name.definition.constructor) @definition.constructor

; Module definitions
(module_declaration
  name: (identifier) @name.definition.module) @definition.module

; Function applications
(application_expression
  function: (identifier) @name.reference.call) @reference.call

; Constructor references
(constructor_pattern
  constructor: (identifier) @name.reference.constructor) @reference.constructor

; Type references
(type_identifier) @name.reference.type @reference.type

; Module references
(qualified_identifier
  module: (identifier) @name.reference.module) @reference.module

; Pattern matching
(pattern
  (identifier) @name.definition.variable) @definition.variable

; Let bindings
(let_expression
  binding: (identifier) @name.definition.variable) @definition.variable

"""
        return template

    def _generate_shell_template(self, lang_name: str, config: Dict) -> str:
        template = f"; {lang_name.title()} shell language tree-sitter tags query file\n\n"

        template += """
; Function definitions
(function_definition
  name: (identifier) @name.definition.function) @definition.function

; Variable assignments
(variable_assignment
  name: (identifier) @name.definition.variable) @definition.variable

; Command calls
(command
  name: (command_name) @name.reference.call) @reference.call

; Variable expansions
(variable_expansion
  name: (variable_name) @name.reference.variable) @reference.variable

; Command substitutions
(command_substitution
  (command
    name: (command_name) @name.reference.call)) @reference.call

; Environment variables
(variable_assignment
  name: (variable_name) @name.definition.variable) @definition.variable

; Aliases
(alias_statement
  name: (identifier) @name.definition.alias) @definition.alias

; Script arguments
(special_variable
  (#match? @special_variable "^[$][0-9@*#?$!-]")) @name.reference.argument

"""
        return template

    def
