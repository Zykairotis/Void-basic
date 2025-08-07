; Wasm language tree-sitter tags query file
; Language type: binary
; Extensions: .wasm, .wat

; Function definitions
(func
+  name: (identifier) @name.definition.function) @definition.function

; Function calls
(call_expression
+  function: (identifier) @name.reference.call) @reference.call

