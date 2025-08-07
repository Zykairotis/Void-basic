; ASM assembly language tree-sitter tags query file

; Label definitions (functions and data labels)
(label
  name: (identifier) @name.definition.function) @definition.function

; Global symbol definitions
(directive
  name: (directive_name) @directive_type
  (identifier) @name.definition.function
  (#eq? @directive_type ".global") @definition.function

(directive
  name: (directive_name) @directive_type
  (identifier) @name.definition.function
  (#eq? @directive_type ".globl") @definition.function

; External symbol declarations
(directive
  name: (directive_name) @directive_type
  (identifier) @name.reference.function
  (#eq? @directive_type ".extern") @reference.function

; Section directives
(directive
  name: (directive_name) @directive_type
  (#any-of? @directive_type ".text" ".data" ".bss" ".rodata") @definition.section

; Named sections
(directive
  name: (directive_name) @directive_type
  (string) @name.definition.section
  (#eq? @directive_type ".section") @definition.section

; Function calls
(instruction
  mnemonic: (mnemonic) @instruction_type
  (identifier) @name.reference.call
  (#any-of? @instruction_type "call" "jmp" "je" "jne" "jz" "jnz" "jl" "jg" "jle" "jge") @reference.call

; Macro definitions
(directive
  name: (directive_name) @directive_type
  (identifier) @name.definition.macro
  (#eq? @directive_type ".macro") @definition.macro

; Macro calls
(instruction
  mnemonic: (identifier) @name.reference.macro) @reference.macro

; Constant definitions
(directive
  name: (directive_name) @directive_type
  (identifier) @name.definition.constant
  (#any-of? @directive_type ".equ" ".set" ".define") @definition.constant

; Type definitions
(directive
  name: (directive_name) @directive_type
  (identifier) @name.definition.type
  (#any-of? @directive_type ".type" ".size") @definition.type

; Symbol references in operands
(operand
  (identifier) @name.reference.variable) @reference.variable

; Memory references
(memory_operand
  (identifier) @name.reference.variable) @reference.variable

; Immediate operand symbols
(immediate_operand
  (identifier) @name.reference.constant) @reference.constant

; Offset references
(offset_operand
  (identifier) @name.reference.variable) @reference.variable

; Register operands (for completeness)
(register) @name.reference.register

; String constants
(directive
  name: (directive_name) @directive_type
  (string) @name.definition.string
  (#any-of? @directive_type ".ascii" ".asciz" ".string") @definition.string

; Data definitions
(directive
  name: (directive_name) @directive_type
  (#any-of? @directive_type ".byte" ".word" ".long" ".quad" ".double" ".float") @definition.data

; Alignment directives
(directive
  name: (directive_name) @directive_type
  (#any-of? @directive_type ".align" ".balign" ".p2align") @definition.alignment

; Include files
(directive
  name: (directive_name) @directive_type
  (string) @name.reference.include
  (#eq? @directive_type ".include") @reference.include

; Conditional assembly
(directive
  name: (directive_name) @directive_type
  (#any-of? @directive_type ".ifdef" ".ifndef" ".if" ".else" ".endif") @definition.conditional

; Local labels (numbered labels like 1:, 2:, etc.)
(local_label
  (number) @name.definition.local) @definition.local

; Local label references (1f, 1b, etc.)
(operand
  (local_label_ref) @name.reference.local) @reference.local

; Procedure definitions (for MASM/NASM style)
(directive
  name: (directive_name) @directive_type
  (identifier) @name.definition.function
  (#any-of? @directive_type "PROC" "proc") @definition.function

; Procedure end
(directive
  name: (directive_name) @directive_type
  (identifier) @name.reference.function
  (#any-of? @directive_type "ENDP" "endp") @reference.function

; Structure definitions (MASM/NASM)
(directive
  name: (directive_name) @directive_type
  (identifier) @name.definition.struct
  (#any-of? @directive_type "STRUCT" "struct" "STRUC" "struc") @definition.struct

; Structure field access
(operand
  (identifier) @struct_name
  "."
  (identifier) @name.reference.field) @reference.field
