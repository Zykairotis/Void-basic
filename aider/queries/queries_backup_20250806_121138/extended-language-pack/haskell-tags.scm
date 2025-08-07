; Haskell language tree-sitter tags query file

; Module definitions
(module_declaration
  name: (module_name) @name.definition.module) @definition.module

; Function definitions
(function_declaration
  name: (variable) @name.definition.function) @definition.function

(function_binding
  name: (variable) @name.definition.function) @definition.function

; Pattern function definitions
(function_lhs
  name: (variable) @name.definition.function) @definition.function

; Type signatures
(type_signature
  name: (variable) @name.definition.function) @definition.function

; Data type definitions
(data_declaration
  name: (type_constructor) @name.definition.type) @definition.type

; Data constructors
(data_constructor_declaration
  name: (constructor) @name.definition.constructor) @definition.constructor

; Type aliases
(type_declaration
  name: (type_constructor) @name.definition.type) @definition.type

; Newtype definitions
(newtype_declaration
  name: (type_constructor) @name.definition.type) @definition.type

; Type class definitions
(class_declaration
  name: (type_class) @name.definition.class) @definition.class

; Type class instance definitions
(instance_declaration
  class: (type_class) @name.reference.class) @reference.class

; Record field definitions
(field_declaration
  name: (variable) @name.definition.field) @definition.field

; Import statements
(import_declaration
  module: (module_name) @name.reference.module) @reference.module

; Qualified imports
(qualified_import_declaration
  module: (module_name) @name.reference.module) @reference.module

; Import aliases
(import_alias
  alias: (module_name) @name.definition.alias) @definition.alias

; Export specifications
(export_specification
  (variable) @name.reference.export) @reference.export

; Function applications
(function_application
  function: (variable) @name.reference.call) @reference.call

; Constructor applications
(constructor_application
  constructor: (constructor) @name.reference.constructor) @reference.constructor

; Variable references
(variable) @name.reference.variable @reference.variable

; Constructor references
(constructor) @name.reference.constructor @reference.constructor

; Type constructor references
(type_constructor) @name.reference.type @reference.type

; Type class references
(type_class) @name.reference.class @reference.class

; Module qualified references
(qualified_variable
  module: (module_name) @name.reference.module
  name: (variable) @name.reference.call) @reference.call

(qualified_constructor
  module: (module_name) @name.reference.module
  name: (constructor) @name.reference.constructor) @reference.constructor

(qualified_type_constructor
  module: (module_name) @name.reference.module
  name: (type_constructor) @name.reference.type) @reference.type

; Let bindings
(let_expression
  (let_binding
    name: (variable) @name.definition.variable)) @definition.variable

; Where bindings
(where_clause
  (function_binding
    name: (variable) @name.definition.function)) @definition.function

(where_clause
  (pattern_binding
    pattern: (variable) @name.definition.variable)) @definition.variable

; Lambda expressions
(lambda_expression
  parameter: (variable) @name.definition.parameter) @definition.parameter

; Case expressions
(case_expression
  (alternative
    pattern: (constructor_pattern
      constructor: (constructor) @name.reference.constructor))) @reference.constructor

; Pattern matching variables
(pattern
  (variable) @name.definition.variable) @definition.variable

; As patterns
(as_pattern
  alias: (variable) @name.definition.variable) @definition.variable

; List comprehension generators
(generator
  pattern: (variable) @name.definition.variable) @definition.variable

; Type annotations in expressions
(type_annotation
  type: (type_constructor) @name.reference.type) @reference.type

; Kind signatures
(kind_signature
  type: (type_constructor) @name.reference.type) @reference.type

; Foreign function interface
(foreign_declaration
  name: (variable) @name.definition.foreign) @definition.foreign

; Default declarations
(default_declaration
  type: (type_constructor) @name.reference.type) @reference.type

; Deriving clauses
(deriving_clause
  class: (type_class) @name.reference.class) @reference.class

; Associated type declarations
(type_family_declaration
  name: (type_constructor) @name.definition.type_family) @definition.type_family

; Data family declarations
(data_family_declaration
  name: (type_constructor) @name.definition.data_family) @definition.data_family

; Type instance declarations
(type_instance_declaration
  name: (type_constructor) @name.reference.type) @reference.type

; Data instance declarations
(data_instance_declaration
  name: (type_constructor) @name.reference.type) @reference.type

; GADT constructors
(gadt_constructor
  name: (constructor) @name.definition.constructor) @definition.constructor

; Record updates
(record_update
  field: (variable) @name.reference.field) @reference.field

; Record construction
(record_construction
  field: (variable) @name.reference.field) @reference.field

; Field access
(field_access
  field: (variable) @name.reference.field) @reference.field

; Operator definitions
(operator_declaration
  operator: (operator) @name.definition.operator) @definition.operator

; Operator references
(operator) @name.reference.operator @reference.operator

; Infix declarations
(infix_declaration
  operator: (operator) @name.reference.operator) @reference.operator

; Pragma annotations
(pragma
  name: (identifier) @name.reference.pragma) @reference.pragma

; Language extensions
(language_pragma
  extension: (identifier) @name.reference.extension) @reference.extension

; Template Haskell splices
(splice_expression
  (variable) @name.reference.splice) @reference.splice

; QuasiQuote references
(quasiquote
  quoter: (variable) @name.reference.quoter) @reference.quoter

; Type holes
(type_hole) @name.reference.hole @reference.hole

; Wildcard patterns
(wildcard_pattern) @reference.wildcard

; Irrefutable patterns
(irrefutable_pattern
  pattern: (variable) @name.definition.variable) @definition.variable

; Bang patterns (strict evaluation)
(bang_pattern
  pattern: (variable) @name.definition.variable) @definition.variable
