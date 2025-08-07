# Extended Language Pack for Tree-sitter Queries

This directory contains comprehensive tree-sitter query files (`.scm`) for over 200 programming languages, markup languages, configuration formats, and domain-specific languages. These queries enable advanced code navigation, semantic highlighting, and structural analysis across a vast ecosystem of languages.

## Overview

Tree-sitter queries use S-expression syntax to match patterns in syntax trees and extract semantic information. This extended language pack provides:

- **Function/method definitions and calls**
- **Class, interface, and type definitions** 
- **Variable and constant declarations**
- **Module/namespace organization**
- **Import/export statements**
- **Language-specific constructs** (annotations, decorators, directives, etc.)

## Supported Languages

### Systems Programming
- **Ada** - Military/aerospace systems language with strong typing
- **Apex** - Salesforce cloud development platform language
- **ASM** - Assembly language for various architectures
- **C/C++** - Low-level systems programming languages
- **Carbon** - Google's experimental C++ successor
- **Crystal** - Ruby-like syntax with static typing
- **D** - Systems programming with garbage collection
- **Nim** - Python-like syntax with C performance
- **Odin** - Simple, fast, data-oriented language
- **Rust** - Memory-safe systems programming
- **V** - Simple, fast, compiled language
- **Zig** - Simple, optimal, and reusable systems programming

### Functional Programming
- **Agda** - Dependently typed functional language
- **Clojure** - Lisp dialect on the JVM
- **Elm** - Functional language for reliable web apps
- **Elixir** - Dynamic, functional language for maintainable applications
- **F#** - Functional-first programming on .NET
- **Haskell** - Pure functional programming language
- **Idris** - Dependently typed functional language
- **Lean** - Theorem prover and programming language
- **OCaml** - Industrial-strength functional programming
- **PureScript** - Strongly-typed functional programming for the web
- **Reason** - Syntax extension for OCaml

### Web Technologies
- **Angular** - TypeScript-based web application framework templates
- **Astro** - Modern static site generator with component islands
- **CSS/SCSS/Less/Stylus** - Stylesheet languages and preprocessors
- **HTML** - Hypertext markup language
- **JavaScript/TypeScript** - Web scripting languages
- **JSX/TSX** - React component syntax extensions
- **Svelte** - Cybernetically enhanced web apps
- **Vue** - Progressive framework for user interfaces

### Mobile Development
- **Dart** - Client-optimized language for mobile/web/desktop
- **Kotlin** - Modern programming language for Android
- **Swift** - Powerful language for iOS/macOS development
- **Objective-C** - Object-oriented language for Apple platforms

### JVM Languages
- **Groovy** - Dynamic language for the Java platform
- **Java** - Object-oriented programming language
- **Scala** - Object-functional programming on the JVM

### Scripting Languages
- **AWK** - Pattern scanning and processing language
- **Bash** - Unix shell and command language
- **Fish** - Smart and user-friendly command line shell
- **Lua** - Lightweight, embeddable scripting language
- **Perl** - High-level, interpreted dynamic programming language
- **PowerShell** - Cross-platform command-line shell and scripting language
- **Python** - High-level programming language
- **Ruby** - Dynamic, open source programming language
- **R** - Statistical computing and graphics language
- **Tcl** - Tool command language

### Data Science & Scientific Computing
- **Julia** - High-performance dynamic language for technical computing
- **MATLAB** - Multi-paradigm numerical computing environment
- **R** - Statistical analysis and data visualization

### Configuration & Data Formats
- **JSON/JSON5/JSONC** - JavaScript Object Notation variants
- **TOML** - Tom's Obvious Minimal Language
- **YAML** - YAML Ain't Markup Language
- **XML** - Extensible Markup Language
- **INI** - Configuration file format
- **Properties** - Java properties files

### Database Languages
- **SQL** - Structured Query Language
- **GraphQL** - Query language for APIs
- **SPARQL** - SPARQL Protocol and RDF Query Language

### Game Development & Graphics
- **GDScript** - Godot's built-in scripting language
- **GLSL** - OpenGL Shading Language
- **HLSL** - High-Level Shading Language
- **WGSL** - WebGPU Shading Language

### Documentation
- **Markdown** - Lightweight markup language
- **AsciiDoc** - Text document format
- **reStructuredText** - Markup syntax and parser system
- **Org-mode** - Document editing and organizing mode

### Infrastructure as Code
- **Terraform** - Infrastructure as code software tool
- **Ansible** - IT automation platform
- **Kubernetes** - Container orchestration manifests
- **Docker** - Containerization platform files

### Protocol Definition
- **Protocol Buffers** - Google's language-neutral data serialization
- **Apache Thrift** - Cross-language services development
- **Apache Avro** - Data serialization system

## Query File Structure

Each `.scm` file follows a consistent pattern:

```scheme
; Language name tree-sitter tags query file

; Definitions (items being declared)
(definition_pattern
  name: (identifier) @name.definition.category) @definition.category

; References (items being used)
(reference_pattern
  name: (identifier) @name.reference.category) @reference.category
```

### Standard Categories

- **`@definition.function`** - Function/method/procedure definitions
- **`@definition.class`** - Class, struct, interface definitions  
- **`@definition.type`** - Type aliases, custom types
- **`@definition.variable`** - Variable declarations
- **`@definition.constant`** - Constant definitions
- **`@definition.module`** - Module/namespace definitions
- **`@definition.interface`** - Interface definitions

- **`@reference.call`** - Function/method calls
- **`@reference.class`** - Type/class references
- **`@reference.variable`** - Variable usage
- **`@reference.module`** - Module/import references

### Language-Specific Extensions

Many languages include specialized categories:

- **`@definition.constructor`** - Data type constructors (Haskell, F#)
- **`@definition.macro`** - Macro definitions (Rust, C)
- **`@definition.trait`** - Trait definitions (Rust, Scala)
- **`@definition.protocol`** - Protocol definitions (Swift)
- **`@definition.directive`** - Framework directives (Angular, Vue)
- **`@definition.annotation`** - Language annotations (Java, C#)

## Usage Examples

### Integration with Code Editors

These query files can be used with any editor supporting tree-sitter:

**Neovim (nvim-treesitter)**:
```lua
require'nvim-treesitter.configs'.setup {
  ensure_installed = "all",
  highlight = { enable = true },
  textobjects = { enable = true }
}
```

**Emacs (tree-sitter)**:
```elisp
(add-to-list 'tree-sitter-load-path "/path/to/extended-language-pack")
```

**Helix Editor**:
```toml
# Add to languages.toml
[[language]]
name = "your-language"
scope = "source.your-language"
roots = []
```

### Code Analysis Tools

These queries enable building powerful code analysis tools:

```python
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

# Load language and query
PY_LANGUAGE = Language(tspython.language(), "python")
query = PY_LANGUAGE.query("""
  (function_definition
    name: (identifier) @name.definition.function) @definition.function
""")

# Parse and extract functions
parser = Parser()
parser.set_language(PY_LANGUAGE)
tree = parser.parse(bytes(source_code, "utf8"))
captures = query.captures(tree.root_node)
```

## Contributing

### Adding New Languages

1. **Create the query file**: `{language}-tags.scm`
2. **Follow naming conventions**: Use standard capture categories
3. **Include language-specific patterns**: Add unique constructs
4. **Add comprehensive coverage**: Include all major language features
5. **Test thoroughly**: Verify queries work with real code

### Query Pattern Guidelines

```scheme
; Use descriptive comments
; Group related patterns together
; Include both common and language-specific patterns
; Use predicates for pattern refinement (#match?, #eq?, etc.)
; Consider performance implications of complex patterns
```

### Language Research Template

When adding a new language:

1. **Study the grammar**: Understand AST node types
2. **Identify key constructs**: Functions, classes, variables, etc.
3. **Find language-specific features**: Unique syntax patterns
4. **Test with real code**: Use actual project files
5. **Document extensions**: Note any custom capture categories

## Advanced Features

### Predicate Usage

Tree-sitter queries support powerful predicates:

```scheme
; Match specific patterns
(function_declaration
  name: (identifier) @name.definition.function
  (#match? @name.definition.function "^test_"))

; Exclude certain matches  
(call_expression
  function: (identifier) @name.reference.call
  (#not-eq? @name.reference.call "print"))

; Multiple conditions
(variable_assignment
  name: (identifier) @name.definition.variable
  (#match? @name.definition.variable "^[A-Z_]+$")) ; Constants
```

### Multi-Language Support

Some files support multiple related languages:

- **`javascript-tags.scm`** - Also handles JSX patterns
- **`typescript-tags.scm`** - Extends JavaScript with types
- **`c-tags.scm`** - Base patterns for C-family languages
- **`shell-tags.scm`** - Common patterns for shell languages

### Context-Aware Patterns

Advanced queries consider context:

```scheme
; Method definitions vs function definitions
(class_declaration
  body: (class_body
    (method_definition
      name: (identifier) @name.definition.method))) @definition.method

; Module-level functions
(module
  (function_declaration
    name: (identifier) @name.definition.function)) @definition.function
```

## Performance Considerations

- **Avoid overly broad patterns** - Be specific to reduce false matches
- **Use predicates wisely** - Complex regex predicates can be slow
- **Group similar patterns** - Organize queries for efficient parsing
- **Test on large files** - Ensure queries perform well at scale

## License & Credits

This extended language pack builds upon the excellent work of:

- [Tree-sitter organization](https://github.com/tree-sitter) - Core parsers and tooling
- [nvim-treesitter](https://github.com/nvim-treesitter/nvim-treesitter) - Neovim integration and queries
- [Helix Editor](https://github.com/helix-editor/helix) - Advanced editor queries
- Various language communities - Parser development and maintenance

Many query patterns are adapted from official tree-sitter language repositories under their respective open-source licenses (primarily MIT and Apache 2.0).

## Future Roadmap

- **Expand to 300+ languages** - Add more niche and emerging languages
- **Semantic analysis queries** - Beyond basic navigation
- **Language server integration** - Enable LSP-style features
- **Cross-language patterns** - Polyglot project support
- **Performance optimization** - Faster query execution
- **Documentation generation** - Automatic API docs from queries

## Getting Started

1. **Choose your integration**: Editor plugin, analysis tool, or custom application
2. **Load the language parsers**: Install tree-sitter bindings
3. **Import query files**: Point to this directory
4. **Test with sample code**: Verify patterns work as expected
5. **Customize as needed**: Extend or modify patterns for your use case

For detailed integration guides, see the `examples/` directory and language-specific documentation.