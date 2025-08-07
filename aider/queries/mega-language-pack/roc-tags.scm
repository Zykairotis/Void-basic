; Tree-sitter query file for Roc
; Language: Roc
; Type: Functional
; Description: Fast, friendly, functional language
; Version: 1.0
; Generated: 2025-08-06
; Features: applications, platforms, abilities, types


; Function definitions
(function_definition
  name: (identifier) @name.definition.function) @definition.function

(function_declaration
  name: (identifier) @name.definition.function) @definition.function

; Function calls
(call_expression
  function: (identifier) @name.reference.call) @reference.call

(function_call
  name: (identifier) @name.reference.call) @reference.call

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

; Comments
(comment) @comment

; Strings
(string) @string

; Numbers
(number) @number
(integer) @number
(float) @number
