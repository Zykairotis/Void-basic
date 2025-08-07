; Capnp language tree-sitter tags query file
; Language type: protocol
; Extensions: .capnp

; Struct definitions
(struct_definition
+  name: (identifier) @name.definition.class) @definition.class

; Struct field definitions
(field_declaration
+  name: (identifier) @name.definition.field) @definition.field

; Enum definitions
(enum_definition
+  name: (identifier) @name.definition.enum) @definition.enum

; Enum constant definitions
(enum_constant
+  name: (identifier) @name.definition.constant) @definition.constant

; Interface definitions
(interface_definition
+  name: (identifier) @name.definition.interface) @definition.interface

; Interface implementations
(interface_implementation
+  name: (identifier) @name.reference.interface) @reference.interface

; Message definitions
(message_definition
+  name: (identifier) @name.definition.message) @definition.message

; Field definitions
(field_definition
+  name: (identifier) @name.definition.field) @definition.field

; Service definitions
(service_definition
+  name: (identifier) @name.definition.service) @definition.service

; Method definitions in services
(method_definition
+  name: (identifier) @name.definition.method) @definition.method

