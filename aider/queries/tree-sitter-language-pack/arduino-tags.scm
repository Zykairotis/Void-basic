; Tree-sitter query file for Arduino
; Language: Arduino
; Version: 1.0
; Generated: 2025-08-06

(function_declarator
  declarator: (identifier) @name.definition.function) @definition.function

(call_expression
  function: (identifier) @name.reference.call) @reference.call
