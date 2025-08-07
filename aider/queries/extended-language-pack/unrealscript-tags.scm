; Unrealscript language tree-sitter tags query file
; Language type: game
; Extensions: .uc

; Function definitions
(function_declaration
+  name: (identifier) @name.definition.function) @definition.function

; Function calls
(call_expression
+  function: (identifier) @name.reference.call) @reference.call

; Class definitions
(class_declaration
+  name: (identifier) @name.definition.class) @definition.class

; Class references
(type_identifier) @name.reference.class @reference.class

