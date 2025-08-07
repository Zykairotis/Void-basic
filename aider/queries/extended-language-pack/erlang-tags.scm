; Erlang language tree-sitter tags query file
; Language type: functional
; Extensions: .erl, .hrl

; Function definitions
(function_clause
+  name: (identifier) @name.definition.function) @definition.function

; Function calls
(call_expression
+  function: (identifier) @name.reference.call) @reference.call

; Module definitions
(module_attribute
+  name: (identifier) @name.definition.module) @definition.module

; Module references
(import_statement
+  (identifier) @name.reference.module) @reference.module

