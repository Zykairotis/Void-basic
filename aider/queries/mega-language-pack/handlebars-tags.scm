; Tree-sitter query file for Handlebars
; Language: Handlebars
; Type: Template
; Description: JavaScript template engine
; Version: 1.0
; Generated: 2025-08-06
; Features: helpers, partials, blocks, variables


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

; Template variables
(variable) @name.reference.variable @reference.variable

; Template blocks
(block_statement
  name: (identifier) @name.definition.block) @definition.block

; Template filters
(filter_expression
  filter: (identifier) @name.reference.filter) @reference.filter

; Comments
(comment) @comment

; Strings
(string) @string

; Numbers
(number) @number
(integer) @number
(float) @number
