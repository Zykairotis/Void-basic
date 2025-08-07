; Tree-sitter query file for Tcsh
; Language: Tcsh
; Type: Shell
; Description: C shell with enhancements
; Version: 1.0
; Generated: 2025-08-06
; Features: aliases, variables, functions, history


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

; Command invocations
(command_invocation
  command: (identifier) @name.reference.command) @reference.command

; Function definitions
(function_definition
  name: (identifier) @name.definition.function) @definition.function

; Alias definitions
(alias_statement
  name: (identifier) @name.definition.alias) @definition.alias

; Comments
(comment) @comment

; Strings
(string) @string

; Numbers
(number) @number
(integer) @number
(float) @number
