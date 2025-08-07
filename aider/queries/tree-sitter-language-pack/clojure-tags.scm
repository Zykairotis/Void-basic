; Tree-sitter query file for Clojure
; Language: Clojure
; Version: 1.0
; Generated: 2025-08-06

(list_lit
  meta: _*
  . (sym_lit name: (sym_name) @ignore)
  . (sym_lit name: (sym_name) @name.definition.method)
  (#match? @ignore "^def.*")

(sym_lit name: (sym_name) @name.reference.call)
