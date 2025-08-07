; Alloy language tree-sitter tags query file
; Language type: specification
; Extensions: .als


; Function definitions
(function
+  name: (identifier) @name.definition.function) @definition.function

; Function calls
(call_expression
+  function: (identifier) @name.reference.call) @reference.call

