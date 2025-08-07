; AWK language tree-sitter tags query file

; Function definitions
(function_definition
  name: (identifier) @name.definition.function) @definition.function

; Variable assignments (global scope)
(assignment_expression
  left: (identifier) @name.definition.variable) @definition.variable

; Array assignments
(assignment_expression
  left: (field_expression
    object: (identifier) @name.definition.array) @definition.array

; Local variable declarations in functions
(function_definition
  body: (block_statement
    (assignment_expression
      left: (identifier) @name.definition.variable)) @definition.variable

; Function calls
(call_expression
  function: (identifier) @name.reference.call) @reference.call

; Built-in function calls
(call_expression
  function: (identifier) @name.reference.builtin
  (#any-of? @name.reference.builtin)
    "print" "printf" "getline" "next" "exit" "return"
    "length" "substr" "index" "split" "sub" "gsub"
    "match" "sprintf" "tolower" "toupper" "system"
    "strftime" "mktime" "systime" "rand" "srand"
    "int" "sqrt" "exp" "log" "sin" "cos" "atan2") @reference.builtin

; Built-in variables
(identifier
  (#any-of? @identifier)
    "NF" "NR" "FNR" "FILENAME" "FS" "RS" "OFS" "ORS"
    "RSTART" "RLENGTH" "SUBSEP" "ARGC" "ARGV" "ENVIRON"
    "ERRNO" "FIELDWIDTHS" "FPAT" "IGNORECASE" "LINT"
    "OFMT" "CONVFMT" "BINMODE" "RT") @name.reference.builtin_variable

; Field references ($0, $1, $2, etc.)
(field_expression
  "$"
  (number_literal) @name.reference.field) @reference.field

; Field references with variables ($NF, $i, etc.)
(field_expression
  "$"
  (identifier) @name.reference.field_var) @reference.field

; Variable references
(identifier) @name.reference.variable @reference.variable

; Array element access
(field_expression
  object: (identifier) @name.reference.array
  field: (identifier) @reference.array

; Pattern definitions (regex patterns)
(regex_pattern) @definition.pattern

; BEGIN block
(rule
  pattern: (identifier) @pattern_type
  (#eq? @pattern_type "BEGIN") @definition.begin_block

; END block
(rule
  pattern: (identifier) @pattern_type
  (#eq? @pattern_type "END") @definition.end_block

; Pattern-action rules
(rule
  pattern: (_) @definition.pattern
  action: (block_statement) @definition.rule

; For loop variables
(for_statement
  initializer: (assignment_expression
    left: (identifier) @name.definition.loop_variable) @definition.variable

; For-in loop variables (for array iteration)
(for_in_statement
  left: (identifier) @name.definition.iterator) @definition.variable

; While loop condition variables
(while_statement
  condition: (identifier) @name.reference.condition) @reference.variable

; If condition variables
(if_statement
  condition: (identifier) @name.reference.condition) @reference.variable

; String literals in patterns
(string_literal) @reference.string

; Regular expression literals
(regex_literal) @reference.regex

; Comparison operators with variables
(binary_expression
  left: (identifier) @name.reference.operand) @reference.variable

(binary_expression
  right: (identifier) @name.reference.operand) @reference.variable

; Unary expressions
(unary_expression
  operand: (identifier) @name.reference.operand) @reference.variable

; Conditional expressions (ternary)
(conditional_expression
  condition: (identifier) @name.reference.condition) @reference.variable

; Member expressions for arrays
(member_expression
  object: (identifier) @name.reference.array) @reference.array

; Post/pre increment operations
(update_expression
  operand: (identifier) @name.reference.variable) @reference.variable

; Delete statement for arrays
(delete_statement
  operand: (field_expression
    object: (identifier) @name.reference.array) @reference.array

; Next statement
(next_statement) @reference.control

; Exit statement
(exit_statement) @reference.control

; Return statement in functions
(return_statement
  (identifier) @name.reference.return_value) @reference.variable

; Parameter references in functions
(function_definition
  parameters: (parameter_list
    (identifier) @name.definition.parameter) @definition.parameter
