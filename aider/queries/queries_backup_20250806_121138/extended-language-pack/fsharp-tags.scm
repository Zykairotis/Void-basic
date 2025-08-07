; Fsharp language tree-sitter tags query file
; Language type: functional
; Extensions: .fs, .fsx, .fsi


; Function definitions
(function_definition
+  name: (identifier) @name.definition.function) @definition.function

; Function calls
(call_expression
+  function: (identifier) @name.reference.call) @reference.call


; Class definitions
(type_definition
+  name: (identifier) @name.definition.class) @definition.class

; Class references
(type_identifier) @name.reference.class @reference.class


; Module definitions
(module_declaration
+  name: (identifier) @name.definition.module) @definition.module

; Module references
(import_statement
+  (identifier) @name.reference.module) @reference.module


; Type definitions
(type_definition
+  name: (identifier) @name.definition.type) @definition.type

; Type references
(type_identifier) @name.reference.type @reference.type

