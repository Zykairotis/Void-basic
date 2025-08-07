; Xquery language tree-sitter tags query file
; Language type: query
; Extensions: .xq, .xquery

; Function definitions
(function_declaration
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

; Type definitions
(type_declaration
+  name: (identifier) @name.definition.type) @definition.type

; Type references
(type_identifier) @name.reference.type @reference.type

