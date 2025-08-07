; Tree-sitter query file for Matlab
; Language: Matlab
; Version: 1.0
; Generated: 2025-08-06

(class_definition
  name: (identifier) @name.definition.class) @definition.class

(function_definition
  name: (identifier) @name.definition.function) @definition.function

(function_call
  name: (identifier) @name.reference.call) @reference.call

(command (command_name) @name.reference.call) @reference.call
