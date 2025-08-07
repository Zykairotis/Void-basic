; Tree-sitter query file for Udev
; Language: Udev
; Version: 1.0
; Generated: 2025-08-06

(assignment
  key: "LABEL"
  (value
    (content) @name.definition.label) @definition.label

(assignment
  key: "GOTO"
  (value
    (content) @name.reference.label) @reference.label

(assignment
  key: "ENV"
  (env_var) @name.definition.variable) @definition.variable

(match
  key: "ENV"
  (env_var) @name.reference.variable) @reference.variable

(var_sub
  (env_var) @name.reference.variable) @reference.variable
