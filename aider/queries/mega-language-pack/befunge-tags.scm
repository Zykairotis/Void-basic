; Tree-sitter query file for Befunge
; Language: Befunge
; Type: Esoteric
; Description: Two-dimensional programming
; Version: 1.0
; Generated: 2025-08-06
; Features: grid, stack, directions, commands


; Language-specific constructs

; Stack patterns
(stack_operation
  name: (identifier) @name.definition.stack) @definition.stack

(stack_operation
  (identifier) @name.reference.stack) @reference.stack
