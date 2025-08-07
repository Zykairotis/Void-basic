; Tree-sitter query file for Cadence
; Language: Cadence
; Type: Blockchain
; Description: Flow blockchain smart contract language
; Version: 1.0
; Generated: 2025-08-06
; Features: contracts, transactions, scripts, resources


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
