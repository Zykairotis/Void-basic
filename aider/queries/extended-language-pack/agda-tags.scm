; Agda language tree-sitter tags query file

; Module definitions
(module_declaration
  name: (identifier) @name.definition.module) @definition.module

; Data type definitions
(data_declaration
  name: (identifier) @name.definition.type) @definition.type

; Data constructors
(constructor_declaration
  name: (identifier) @name.definition.constructor) @definition.constructor

; Record definitions
(record_declaration
  name: (identifier) @name.definition.type) @definition.type

; Record fields
(field_declaration
  name: (identifier) @name.definition.field) @definition.field

; Function definitions
(function_clause
  name: (identifier) @name.definition.function) @definition.function

(function_declaration
  name: (identifier) @name.definition.function) @definition.function

; Type signatures
(type_signature
  name: (identifier) @name.definition.function) @definition.function

; Postulates (axioms)
(postulate_declaration
  name: (identifier) @name.definition.function) @definition.function

; Variable declarations
(variable_declaration
  (identifier) @name.definition.variable) @definition.variable

; Primitive definitions
(primitive_declaration
  name: (identifier) @name.definition.function) @definition.function

; Open statements (imports)
(open_statement
  module: (identifier) @name.reference.module) @reference.module

; Import statements
(import_statement
  module: (qualified_name
    (identifier) @name.reference.module) @reference.module

; Function calls and references
(identifier) @name.reference.call @reference.call

; Constructor references
(qualified_name
  (identifier) @name.reference.constructor) @reference.constructor

; Type references in signatures
(arrow_type
  left: (identifier) @name.reference.type) @reference.type

(arrow_type
  right: (identifier) @name.reference.type) @reference.type

; Application expressions (function calls)
(application
  function: (identifier) @name.reference.call) @reference.call

; Pattern matching on constructors
(constructor_pattern
  constructor: (identifier) @name.reference.constructor) @reference.constructor

; Let bindings
(let_expression
  (let_binding
    name: (identifier) @name.definition.variable) @definition.variable

; Where clauses
(where_clause
  (function_clause
    name: (identifier) @name.definition.function) @definition.function

; Lambda abstractions
(lambda_expression
  parameter: (identifier) @name.definition.variable) @definition.variable

; Forall quantification
(pi_type
  parameter: (identifier) @name.definition.variable) @definition.variable

; Pattern variables
(identifier_pattern
  (identifier) @name.definition.variable) @definition.variable

; Mutual blocks
(mutual_block
  (data_declaration
    name: (identifier) @name.definition.type) @definition.type

(mutual_block
  (function_clause
    name: (identifier) @name.definition.function) @definition.function

; Abstract definitions
(abstract_block
  (data_declaration
    name: (identifier) @name.definition.type) @definition.type

(abstract_block
  (function_clause
    name: (identifier) @name.definition.function) @definition.function

; Private definitions
(private_block
  (data_declaration
    name: (identifier) @name.definition.type) @definition.type

(private_block
  (function_clause
    name: (identifier) @name.definition.function) @definition.function

; Instance definitions
(instance_block
  (function_clause
    name: (identifier) @name.definition.function) @definition.function

; Macro definitions
(macro_declaration
  name: (identifier) @name.definition.function) @definition.function

; Unquoting
(unquote_declaration
  name: (identifier) @name.definition.function) @definition.function
