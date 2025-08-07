; Tree-sitter query file for Redis
; Language: Redis
; Type: Database
; Description: Redis commands and scripts
; Version: 1.0
; Generated: 2025-08-06
; Features: commands, keys, values, scripts


; Variable definitions
(variable_declaration
  name: (identifier) @name.definition.variable) @definition.variable

(assignment_expression
  left: (identifier) @name.definition.variable) @definition.variable

; Variable references
(identifier) @name.reference.variable @reference.variable

; Type definitions
(type_definition
  name: (identifier) @name.definition.type) @definition.type

(type_declaration
  name: (identifier) @name.definition.type) @definition.type

; Language-specific constructs

; Table references
(table_reference
  name: (identifier) @name.reference.table) @reference.table

; Column references
(column_reference
  name: (identifier) @name.reference.column) @reference.column

; Function calls
(function_call
  name: (identifier) @name.reference.function) @reference.function

; Comments
(comment) @comment

; Strings
(string) @string

; Numbers
(number) @number
(integer) @number
(float) @number
