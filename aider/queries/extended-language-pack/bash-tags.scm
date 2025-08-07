; Bash shell language tree-sitter tags query file

; Function definitions
(function_definition
  name: (word) @name.definition.function) @definition.function

; Alternative function syntax (function keyword)
(function_definition
  name: (word) @name.definition.function) @definition.function

; Variable assignments
(variable_assignment
  name: (variable_name) @name.definition.variable) @definition.variable

; Local variable declarations
(declaration_command
  (simple_command
    (command_name) @command_type
    (word) @name.definition.variable
  (#eq? @command_type "local") @definition.variable

; Export declarations
(declaration_command
  (simple_command
    (command_name) @command_type
    (word) @name.definition.variable
  (#eq? @command_type "export") @definition.variable

; Readonly declarations
(declaration_command
  (simple_command
    (command_name) @command_type
    (word) @name.definition.variable
  (#eq? @command_type "readonly") @definition.variable

; Command calls
(command
  name: (command_name) @name.reference.call) @reference.call

(simple_command
  (command_name) @name.reference.call) @reference.call

; Built-in commands
(simple_command
  (command_name) @name.reference.builtin
  (#any-of? @name.reference.builtin)
    "cd" "pwd" "echo" "printf" "read" "test" "exit" "return"
    "source" "." "exec" "eval" "set" "unset" "shift" "getopts"
    "trap" "kill" "jobs" "bg" "fg" "wait" "type" "which"
    "alias" "unalias" "history" "fc" "hash" "help" "builtin"
    "command" "enable" "logout" "times" "ulimit" "umask") @reference.builtin

; Variable expansions
(variable_expansion
  (variable_name) @name.reference.variable) @reference.variable

; Special variables
(variable_expansion
  (variable_name) @name.reference.special
  (#match? @name.reference.special "^[0-9@*#?$!-]$") @reference.special

; Positional parameters
(variable_expansion
  (variable_name) @name.reference.parameter
  (#match? @name.reference.parameter "^[0-9]+$") @reference.parameter

; Environment variables (common ones)
(variable_expansion
  (variable_name) @name.reference.environment
  (#any-of? @name.reference.environment)
    "PATH" "HOME" "USER" "SHELL" "PWD" "OLDPWD" "TERM" "LANG"
    "LC_ALL" "IFS" "PS1" "PS2" "PS3" "PS4" "HISTSIZE" "HISTFILE"
    "EDITOR" "VISUAL" "PAGER" "LESS" "MORE") @reference.environment

; Command substitutions
(command_substitution
  (command
    name: (command_name) @name.reference.call) @reference.call

; Process substitutions
(process_substitution
  (command
    name: (command_name) @name.reference.call) @reference.call

; Array assignments
(variable_assignment
  name: (variable_name) @name.definition.array
  value: (array) @definition.array

; Array element references
(subscript
  name: (variable_name) @name.reference.array) @reference.array

; Associative array assignments
(variable_assignment
  name: (variable_name) @name.definition.hash) @definition.hash

; For loop variables
(for_statement
  variable: (variable_name) @name.definition.variable) @definition.variable

; While loop condition variables
(while_statement
  condition: (test_command
    (variable_expansion
      (variable_name) @name.reference.variable)) @reference.variable

; If condition variables
(if_statement
  condition: (test_command
    (variable_expansion
      (variable_name) @name.reference.variable)) @reference.variable

; Case statement variables
(case_statement
  value: (variable_expansion
    (variable_name) @name.reference.variable) @reference.variable

; Function calls
(simple_command
  (command_name) @name.reference.call
  (#is-not? local) @reference.call

; Source/include statements
(simple_command
  (command_name) @source_cmd
  (word) @name.reference.source
  (#any-of? @source_cmd "source" ".") @reference.source

; Alias definitions
(simple_command
  (command_name) @alias_cmd
  (word) @alias_def
  (#eq? @alias_cmd "alias")
  (#contains? @alias_def "=") @definition.alias

; Here documents
(heredoc_redirect
  (heredoc_start) @name.definition.heredoc) @definition.heredoc

; Redirections with file descriptors
(file_redirect
  descriptor: (file_descriptor) @name.reference.fd) @reference.fd

; Pipeline commands
(pipeline
  (command
    name: (command_name) @name.reference.call) @reference.call

; Subshell commands
(subshell
  (command
    name: (command_name) @name.reference.call) @reference.call

; Test expressions
(test_command
  (word) @name.reference.test_arg) @reference.test

; Arithmetic expansions
(arithmetic_expansion
  (variable_name) @name.reference.variable) @reference.variable

; Parameter expansions
(expansion
  (variable_name) @name.reference.variable) @reference.variable

; Brace expansions
(brace_expression) @reference.brace

; Glob patterns
(word
  (#match? @word "[*?\\[]") @reference.glob

; String variables in double quotes
(string
  (variable_expansion
    (variable_name) @name.reference.variable) @reference.variable

; Command names in strings (for dynamic execution)
(string
  (command_substitution
    (command
      name: (command_name) @name.reference.call)) @reference.call

; Exit status checks
(variable_expansion
  (variable_name) @exit_status
  (#eq? @exit_status "?") @reference.exit_status

; Process ID references
(variable_expansion
  (variable_name) @process_id
  (#eq? @process_id "$") @reference.process_id

; Background process references
(variable_expansion
  (variable_name) @background_pid
  (#eq? @background_pid "!") @reference.background_pid

; Script name references
(variable_expansion
  (variable_name) @script_name
  (#eq? @script_name "0") @reference.script_name

; Argument count references
(variable_expansion
  (variable_name) @arg_count
  (#eq? @arg_count "#") @reference.arg_count

; All arguments references
(variable_expansion
  (variable_name) @all_args
  (#any-of? @all_args "@" "*") @reference.all_args
