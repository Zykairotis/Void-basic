; Systemd language tree-sitter tags query file
; Language type: config
; Extensions: .service, .socket, .timer


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

