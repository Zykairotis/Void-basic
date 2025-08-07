; Tree-sitter query file for Brainfuck
; Language: Brainfuck
; Type: Esoteric
; Description: Minimalist programming language
; Version: 1.0
; Generated: 2025-08-06
; Features: commands, loops, memory


; Language-specific constructs

; Memory patterns
(memory_operation
  name: (identifier) @name.definition.memory) @definition.memory

(memory_operation
  (identifier) @name.reference.memory) @reference.memory
