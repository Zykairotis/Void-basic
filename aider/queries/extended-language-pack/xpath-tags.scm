; Xpath language tree-sitter tags query file
; Language type: query
; Extensions: .xpath

; Function definitions
(function_call
+  name: (identifier) @name.definition.function) @definition.function

; Function calls
(call_expression
+  function: (identifier) @name.reference.call) @reference.call

