; Move language tree-sitter tags query file
; Language type: blockchain
; Extensions: .move

; Function definitions
(function_definition
+  name: (identifier) @name.definition.function) @definition.function

; Function calls
(call_expression
+  function: (identifier) @name.reference.call) @reference.call

; Module definitions
(module_definition
+  name: (identifier) @name.definition.module) @definition.module

; Module references
(import_statement
+  (identifier) @name.reference.module) @reference.module

; Struct definitions
(struct_definition
+  name: (identifier) @name.definition.class) @definition.class

; Struct field definitions
(field_declaration
+  name: (identifier) @name.definition.field) @definition.field

