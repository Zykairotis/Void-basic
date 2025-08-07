; Mediawiki language tree-sitter tags query file
; Language type: markup
; Extensions: .wiki, .mediawiki

; Block elements
(block_element
+  (identifier) @name.definition.block) @definition.block

; Inline elements
(inline_element
+  (identifier) @name.reference.inline) @reference.inline

; Links and references
(link
+  target: (identifier) @name.reference.link) @reference.link

; Headings
(heading
+  text: (identifier) @name.definition.heading) @definition.heading

