; Nim systems programming language tree-sitter tags query file

; Procedure definitions
(proc_declaration
  name: (identifier) @name.definition.function) @definition.function

; Function definitions (pure procedures)
(func_declaration
  name: (identifier) @name.definition.function) @definition.function

; Method definitions
(method_declaration
  name: (identifier) @name.definition.method) @definition.method

; Template definitions
(template_declaration
  name: (identifier) @name.definition.template) @definition.template

; Macro definitions
(macro_declaration
  name: (identifier) @name.definition.macro) @definition.macro

; Iterator definitions
(iterator_declaration
  name: (identifier) @name.definition.iterator) @definition.iterator

; Converter definitions
(converter_declaration
  name: (identifier) @name.definition.converter) @definition.converter

; Type definitions
(type_section
  (type_def
    name: (identifier) @name.definition.type) @definition.type

; Object type definitions
(object_declaration
  name: (identifier) @name.definition.class) @definition.class

; Tuple type definitions
(tuple_declaration
  name: (identifier) @name.definition.type) @definition.type

; Enum definitions
(enum_declaration
  name: (identifier) @name.definition.enum) @definition.enum

; Enum field definitions
(enum_field
  name: (identifier) @name.definition.constant) @definition.constant

; Distinct type definitions
(distinct_declaration
  name: (identifier) @name.definition.type) @definition.type

; Concept definitions (type classes)
(concept_declaration
  name: (identifier) @name.definition.concept) @definition.concept

; Variable declarations
(var_section
  (var_def
    name: (identifier) @name.definition.variable) @definition.variable

; Let declarations (immutable)
(let_section
  (let_def
    name: (identifier) @name.definition.variable) @definition.variable

; Constant declarations
(const_section
  (const_def
    name: (identifier) @name.definition.constant) @definition.constant

; Field declarations in objects
(field_declaration
  name: (identifier) @name.definition.field) @definition.field

; Parameter declarations
(parameter
  name: (identifier) @name.definition.parameter) @definition.parameter

; Generic parameters
(generic_parameter
  name: (identifier) @name.definition.type_parameter) @definition.type_parameter

; Import statements
(import_statement
  (identifier) @name.reference.module) @reference.module

(import_statement
  (dotted_identifier
    (identifier) @name.reference.module) @reference.module

; Include statements
(include_statement
  (identifier) @name.reference.include) @reference.include

; From import statements
(from_import_statement
  module: (identifier) @name.reference.module
  names: (identifier) @name.reference.import) @reference.import

; Export statements
(export_statement
  (identifier) @name.reference.export) @reference.export

; Procedure calls
(call
  function: (identifier) @name.reference.call) @reference.call

; Method calls
(dot_expression
  right: (identifier) @name.reference.call) @reference.call

; Command syntax calls (without parentheses)
(command
  function: (identifier) @name.reference.call) @reference.call

; Infix operator calls
(infix_expression
  operator: (identifier) @name.reference.operator) @reference.operator

; Prefix operator calls
(prefix_expression
  operator: (identifier) @name.reference.operator) @reference.operator

; Type references
(type_expression
  (identifier) @name.reference.type) @reference.type

; Generic type instantiations
(bracket_expression
  left: (identifier) @name.reference.type) @reference.type

; Object construction
(object_construction
  type: (identifier) @name.reference.class) @reference.class

; Tuple construction
(tuple_construction
  type: (identifier) @name.reference.type) @reference.type

; Field access
(dot_expression
  left: (identifier) @name.reference.variable
  right: (identifier) @name.reference.field) @reference.field

; Variable references
(identifier) @name.reference.variable @reference.variable

; Module qualified references
(dot_expression
  left: (identifier) @name.reference.module
  right: (identifier) @name.reference.call) @reference.call

; Pragma definitions
(pragma_declaration
  name: (identifier) @name.definition.pragma) @definition.pragma

; Pragma usages
(pragma_expression
  (identifier) @name.reference.pragma) @reference.pragma

; Exception definitions
(type_section
  (type_def
    name: (identifier) @name.definition.exception
    type: (object_declaration
      parent: (identifier) @parent_type
      (#match? @parent_type ".*Exception$") @definition.exception

; Exception handling
(try_statement
  (except_branch
    exception_type: (identifier) @name.reference.exception) @reference.exception

; Raise statements
(raise_statement
  exception: (identifier) @name.reference.exception) @reference.exception

; Block labels
(block_statement
  label: (identifier) @name.definition.label) @definition.label

; Break with labels
(break_statement
  label: (identifier) @name.reference.label) @reference.label

; For loop variables
(for_statement
  variables: (identifier) @name.definition.variable) @definition.variable

; While loop conditions
(while_statement
  condition: (identifier) @name.reference.variable) @reference.variable

; If statement conditions
(if_statement
  condition: (identifier) @name.reference.variable) @reference.variable

; Case statement expressions
(case_statement
  expression: (identifier) @name.reference.variable) @reference.variable

; Pattern matching in case branches
(of_branch
  pattern: (identifier) @name.definition.variable) @definition.variable

; String interpolation
(format_string
  (format_expression
    (identifier) @name.reference.variable) @reference.variable

; Array/sequence access
(bracket_expression
  left: (identifier) @name.reference.variable) @reference.variable

; Slice expressions
(slice_expression
  object: (identifier) @name.reference.variable) @reference.variable

; Range expressions
(range_expression
  start: (identifier) @name.reference.variable) @reference.variable

(range_expression
  end: (identifier) @name.reference.variable) @reference.variable

; Assignment targets
(assignment
  left: (identifier) @name.reference.variable) @reference.variable

; Compound assignment operators
(compound_assignment
  left: (identifier) @name.reference.variable) @reference.variable

; Lambda expressions (anonymous procedures)
(lambda_expression
  parameters: (parameter
    name: (identifier) @name.definition.parameter) @definition.parameter

; Closure parameters
(closure_expression
  parameters: (parameter
    name: (identifier) @name.definition.parameter) @definition.parameter

; Result variable (implicit in procedures)
(identifier
  (#eq? @identifier "result") @name.reference.result

; Built-in procedures
(call
  function: (identifier) @name.reference.builtin
  (#any-of? @name.reference.builtin)
    "echo" "print" "println" "write" "writeLine" "readLine"
    "len" "high" "low" "ord" "chr" "inc" "dec" "succ" "pred"
    "min" "max" "abs" "clamp" "swap" "move" "reset" "shallow"
    "deepCopy" "new" "alloc" "alloc0" "allocShared" "dealloc"
    "deallocShared" "realloc" "reallocShared" "sizeof" "alignof"
    "cast" "addr" "unsafeAddr" "pointer" "isNil" "repr" "typeof"
    "compiles" "defined" "declared" "hasCustomPragma") @reference.builtin

; Built-in types
(type_expression
  (identifier) @name.reference.builtin_type
  (#any-of? @name.reference.builtin_type)
    "int" "int8" "int16" "int32" "int64" "uint" "uint8" "uint16" "uint32" "uint64"
    "float" "float32" "float64" "bool" "char" "string" "cstring" "pointer"
    "array" "seq" "set" "tuple" "object" "enum" "range" "proc" "method"
    "iterator" "void" "auto" "static" "typedesc" "untyped" "typed") @reference.builtin_type

; Compile-time expressions
(static_expression
  (identifier) @name.reference.compile_time) @reference.compile_time

; When statements (compile-time conditionals)
(when_statement
  condition: (identifier) @name.reference.compile_time) @reference.compile_time

; Generic constraints
(generic_constraint
  type: (identifier) @name.reference.type) @reference.type

; Mixin statements
(mixin_statement
  (identifier) @name.reference.mixin) @reference.mixin

; Bind statements
(bind_statement
  (identifier) @name.reference.bind) @reference.bind

; Using statements
(using_statement
  (identifier) @name.reference.using) @reference.using

; Defer statements
(defer_statement) @reference.defer

; Discard statements
(discard_statement
  expression: (identifier) @name.reference.discard) @reference.variable

; Emit statements (inline assembly/code)
(emit_statement) @reference.emit

; ASM statements
(asm_statement) @reference.asm
