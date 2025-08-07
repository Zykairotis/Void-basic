; Mathematica language tree-sitter tags query file
; Language type: scientific
; Extensions: .nb, .m, .wl


; Function definitions
(function_definition
+  name: (identifier) @name.definition.function) @definition.function

; Function calls
(call_expression
+  function: (identifier) @name.reference.call) @reference.call


; Module definitions
(module_declaration
+  name: (identifier) @name.definition.module) @definition.module

; Module references
(import_statement
+  (identifier) @name.reference.module) @reference.module


; Variable definitions
(assignment
+  name: (identifier) @name.definition.variable) @definition.variable

; Variable references
(identifier) @name.reference.variable @reference.variable

