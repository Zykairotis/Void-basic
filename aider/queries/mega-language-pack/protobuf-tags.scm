; Tree-sitter query file for Protobuf
; Language: Protobuf
; Type: Protocol
; Description: Protocol Buffers interface definition
; Version: 1.0
; Generated: 2025-08-06
; Features: messages, services, enums, imports


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

; Message definitions
(message_definition
  name: (identifier) @name.definition.message) @definition.message

; Service definitions
(service_definition
  name: (identifier) @name.definition.service) @definition.service

; Field definitions
(field_definition
  name: (identifier) @name.definition.field) @definition.field

; Comments
(comment) @comment

; Strings
(string) @string

; Numbers
(number) @number
(integer) @number
(float) @number
