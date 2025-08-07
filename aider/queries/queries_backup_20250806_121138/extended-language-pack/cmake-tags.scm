; Cmake language tree-sitter tags query file
; Language type: build
; Extensions: CMakeLists.txt, .cmake


; Function definitions
(function_def
+  name: (identifier) @name.definition.function) @definition.function

; Function calls
(call_expression
+  function: (identifier) @name.reference.call) @reference.call


; Variable definitions
(variable_ref
+  name: (identifier) @name.definition.variable) @definition.variable

; Variable references
(identifier) @name.reference.variable @reference.variable

