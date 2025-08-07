; Tree-sitter query file for Ink
; Language: Ink
; Type: Blockchain
; Description: Rust-based smart contracts for Substrate
; Version: 1.0
; Generated: 2025-08-06
; Features: contracts, messages, events, storage


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

; Storage patterns
(storage
  name: (identifier) @name.definition.storage) @definition.storage

(storage
  (identifier) @name.reference.storage) @reference.storage

; Contract definitions
(contract_definition
  name: (identifier) @name.definition.contract) @definition.contract

; Function modifiers
(modifier_definition
  name: (identifier) @name.definition.modifier) @definition.modifier

; Event definitions
(event_definition
  name: (identifier) @name.definition.event) @definition.event

; Comments
(comment) @comment

; Strings
(string) @string

; Numbers
(number) @number
(integer) @number
(float) @number
