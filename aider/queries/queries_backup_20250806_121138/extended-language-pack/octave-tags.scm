; Octave language tree-sitter tags query file
; Language type: scientific
; Extensions: .m


; Function definitions
(function_definition
+  name: (identifier) @name.definition.function) @definition.function

; Function calls
(call_expression
+  function: (identifier) @name.reference.call) @reference.call


; Class definitions
(classdef
+  name: (identifier) @name.definition.class) @definition.class

; Class references
(type_identifier) @name.reference.class @reference.class


; Variable definitions
(assignment
+  name: (identifier) @name.definition.variable) @definition.variable

; Variable references
(identifier) @name.reference.variable @reference.variable

