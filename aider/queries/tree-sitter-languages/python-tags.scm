; Tree-sitter query file for Python
; Language: Python
; Version: 1.0
; Generated: 2025-08-06

(class_definition
  name: (identifier) @name.definition.class) @definition.class

(function_definition
  name: (identifier) @name.definition.function) @definition.function

(call
  function: [
      (identifier) @name.reference.call
      (attribute
        attribute: (identifier) @name.reference.call
  ]) @reference.call
