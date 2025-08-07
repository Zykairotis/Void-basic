; Tree-sitter query file for Nix
; Language: Nix
; Type: Config
; Description: Purely functional package manager
; Version: 1.0
; Generated: 2025-08-06
; Features: functions, derivations, packages, sets


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

; Configuration keys
(property_name) @name.definition.property @definition.property

; Configuration values
(property_value) @reference.value

; Section headers
(section_header) @name.definition.section @definition.section

; Comments
(comment) @comment

; Strings
(string) @string

; Numbers
(number) @number
(integer) @number
(float) @number
