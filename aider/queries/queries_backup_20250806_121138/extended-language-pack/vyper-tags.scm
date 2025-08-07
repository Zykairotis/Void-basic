; Vyper language tree-sitter tags query file
; Language type: blockchain
; Extensions: .vy


; Function definitions
(function_def
+  name: (identifier) @name.definition.function) @definition.function

; Function calls
(call_expression
+  function: (identifier) @name.reference.call) @reference.call


; Struct definitions
(struct_def
+  name: (identifier) @name.definition.class) @definition.class

; Struct field definitions
(field_declaration
+  name: (identifier) @name.definition.field) @definition.field

