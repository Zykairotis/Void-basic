; Bazel language tree-sitter tags query file
; Language type: build
; Extensions: BUILD, WORKSPACE, .bzl

; Function definitions
(function_definition
+  name: (identifier) @name.definition.function) @definition.function

; Function calls
(call_expression
+  function: (identifier) @name.reference.call) @reference.call

