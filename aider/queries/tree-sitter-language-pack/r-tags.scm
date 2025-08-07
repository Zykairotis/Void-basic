; Tree-sitter query file for R
; Language: R
; Version: 1.0
; Generated: 2025-08-06

(binary_operator
    lhs: (identifier) @name.definition.function
    operator: "<-"
    rhs: (function_definition)
) @definition.function

(binary_operator
    lhs: (identifier) @name.definition.function
    operator: "="
    rhs: (function_definition)
) @definition.function

(call
    function: (identifier) @name.reference.call
) @reference.call

(call
    function: (namespace_operator
        rhs: (identifier) @name.reference.call

) @reference.call
