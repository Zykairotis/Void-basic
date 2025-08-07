; Systemverilog language tree-sitter tags query file
; Language type: hardware
; Extensions: .sv, .svh

; Class definitions
(class_declaration
+  name: (identifier) @name.definition.class) @definition.class

; Class references
(type_identifier) @name.reference.class @reference.class

; Module definitions
(module_declaration
+  name: (identifier) @name.definition.module) @definition.module

; Module references
(import_statement
+  (identifier) @name.reference.module) @reference.module

; Interface definitions
(interface_declaration
+  name: (identifier) @name.definition.interface) @definition.interface

; Interface implementations
(interface_implementation
+  name: (identifier) @name.reference.interface) @reference.interface

; Hardware modules
(module_declaration
+  name: (identifier) @name.definition.module) @definition.module

; Port declarations
(port_declaration
+  name: (identifier) @name.definition.port) @definition.port

; Signal declarations
(signal_declaration
+  name: (identifier) @name.definition.signal) @definition.signal

; Wire connections
(wire_assignment
+  target: (identifier) @name.reference.wire) @reference.wire

