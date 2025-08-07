; Tree-sitter query file for Micropython
; Language: Micropython
; Type: Embedded
; Description: Python for microcontrollers
; Version: 1.0
; Generated: 2025-08-06
; Features: classes, functions, hardware, interrupts


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
