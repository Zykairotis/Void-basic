; Tree-sitter query file for Circuitpython
; Language: Circuitpython
; Type: Embedded
; Description: Python for embedded hardware
; Version: 1.0
; Generated: 2025-08-06
; Features: classes, functions, libraries, hardware


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

; Hardware patterns
(attribute
  name: (identifier) @name.definition.hardware) @definition.hardware

(attribute
  (identifier) @name.reference.hardware) @reference.hardware

; Hardware definitions
(pin_definition
  name: (identifier) @name.definition.pin) @definition.pin

; Interrupt handlers
(interrupt_handler
  name: (identifier) @name.definition.interrupt) @definition.interrupt

; Device references
(device_reference
  name: (identifier) @name.reference.device) @reference.device

; Comments
(comment) @comment

; Strings
(string) @string

; Numbers
(number) @number
(integer) @number
(float) @number
