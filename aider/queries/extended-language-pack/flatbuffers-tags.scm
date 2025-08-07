; Flatbuffers language tree-sitter tags query file
; Language type: protocol
; Extensions: .fbs

; Struct definitions
(struct_declaration
+  name: (identifier) @name.definition.class) @definition.class

; Struct field definitions
(field_declaration
+  name: (identifier) @name.definition.field) @definition.field

; Enum definitions
(enum_declaration
+  name: (identifier) @name.definition.enum) @definition.enum

; Enum constant definitions
(enum_constant
+  name: (identifier) @name.definition.constant) @definition.constant

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

