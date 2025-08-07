; Angelscript language tree-sitter tags query file
; Language type: game
; Extensions: .as

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

; Enum definitions
(enum_declaration
+  name: (identifier) @name.definition.enum) @definition.enum

; Enum constant definitions
(enum_constant
+  name: (identifier) @name.definition.constant) @definition.constant

; Interface definitions
(interface_declaration
+  name: (identifier) @name.definition.interface) @definition.interface

; Interface implementations
(interface_implementation
+  name: (identifier) @name.reference.interface) @reference.interface

