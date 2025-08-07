; Nginx language tree-sitter tags query file
; Language type: config
; Extensions: .conf

; Variable definitions
(variable
+  name: (identifier) @name.definition.variable) @definition.variable

; Variable references
(identifier) @name.reference.variable @reference.variable

; Configuration sections
(section_header
+  name: (identifier) @name.definition.section) @definition.section

; Configuration keys
(property_name
+  (identifier) @name.definition.key) @definition.key

; Configuration values
(property_value
+  (identifier) @name.reference.value) @reference.value

; Include/import directives
(include_directive
+  path: (string) @name.reference.include) @reference.include

