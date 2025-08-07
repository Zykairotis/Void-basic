; Tree-sitter query file for Org_Mode
; Language: Org_Mode
; Type: Markup
; Description: Document editing and organizing mode
; Version: 1.0
; Generated: 2025-08-06
; Features: headlines, drawers, blocks, links


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
