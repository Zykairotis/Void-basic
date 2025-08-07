; Tree-sitter query file for Racket
; Language: Racket
; Version: 1.0
; Generated: 2025-08-06

(list
  .
  (symbol) @reference._define
  (#match? @reference._define "^(define|define/contract)$")
  .
  (list
    .
    (symbol) @name.definition.function) @definition.function

(list
  .
  (symbol) @name.reference.call
