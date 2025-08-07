; Tree-sitter query file for Asciidoc
; Language: Asciidoc
; Type: Markup
; Description: Text document format
; Version: 1.0
; Generated: 2025-08-06
; Features: headers, blocks, attributes, macros


; Language-specific constructs

; Headers
(section_title) @name.definition.section @definition.section

; Links
(link
  destination: (link_destination) @name.reference.link) @reference.link

; References
(reference_definition
  label: (reference_label) @name.definition.reference) @definition.reference

; Comments
(comment) @comment

; Strings
(string) @string

; Numbers
(number) @number
(integer) @number
(float) @number
