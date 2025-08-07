; Fish shell language tree-sitter tags query file

; Function definitions
(function_definition
  name: (word) @name.definition.function) @definition.function

; Variable assignments with set command
(command
  name: (command_name) @set_cmd
  argument: (word) @name.definition.variable
  (#eq? @set_cmd "set") @definition.variable

; Local variable assignments
(command
  name: (command_name) @set_cmd
  argument: (option) @local_flag
  argument: (word) @name.definition.variable
  (#eq? @set_cmd "set")
  (#eq? @local_flag "-l") @definition.variable

; Global variable assignments
(command
  name: (command_name) @set_cmd
  argument: (option) @global_flag
  argument: (word) @name.definition.variable
  (#eq? @set_cmd "set")
  (#eq? @global_flag "-g") @definition.variable

; Universal variable assignments
(command
  name: (command_name) @set_cmd
  argument: (option) @universal_flag
  argument: (word) @name.definition.variable
  (#eq? @set_cmd "set")
  (#eq? @universal_flag "-U") @definition.variable

; Export variable assignments
(command
  name: (command_name) @set_cmd
  argument: (option) @export_flag
  argument: (word) @name.definition.variable
  (#eq? @set_cmd "set")
  (#eq? @export_flag "-x") @definition.variable

; Command calls
(command
  name: (command_name) @name.reference.call) @reference.call

; Built-in commands
(command
  name: (command_name) @name.reference.builtin
  (#any-of? @name.reference.builtin)
    "cd" "pwd" "echo" "printf" "read" "test" "exit" "return"
    "source" "eval" "exec" "bg" "fg" "jobs" "disown" "wait"
    "type" "which" "alias" "unalias" "history" "builtin"
    "command" "commandline" "complete" "abbr" "bind" "set_color"
    "fish_config" "fish_prompt" "fish_right_prompt" "fish_greeting"
    "funced" "funcsave" "functions" "help" "isatty" "math"
    "nextd" "prevd" "dirs" "pushd" "popd" "random" "realpath"
    "status" "string" "suspend" "ulimit" "umask" "vared") @reference.builtin

; Variable expansions
(variable_expansion
  (variable_name) @name.reference.variable) @reference.variable

; Special variables
(variable_expansion
  (variable_name) @name.reference.special
  (#any-of? @name.reference.special "argv" "status" "USER" "HOME")
    "PWD" "OLDPWD" "SHLVL" "fish_pid" "last_pid" "version"
    "fish_version" "fish_greeting" "fish_user_paths") @reference.special

; Environment variables
(variable_expansion
  (variable_name) @name.reference.environment
  (#any-of? @name.reference.environment)
    "PATH" "MANPATH" "CDPATH" "TERM" "LANG" "LC_ALL" "LC_CTYPE"
    "EDITOR" "VISUAL" "PAGER" "SHELL" "TMPDIR" "fish_color_normal"
    "fish_color_command" "fish_color_quote" "fish_color_redirection"
    "fish_color_end" "fish_color_error" "fish_color_param"
    "fish_color_comment" "fish_color_match" "fish_color_selection"
    "fish_color_search_match" "fish_color_operator" "fish_color_escape"
    "fish_color_cwd" "fish_color_cwd_root" "fish_color_valid_path"
    "fish_color_autosuggestion" "fish_color_user" "fish_color_host"
    "fish_color_cancel") @reference.environment

; Command substitutions
(command_substitution
  (command
    name: (command_name) @name.reference.call) @reference.call

; Process substitutions
(process_substitution
  (command
    name: (command_name) @name.reference.call) @reference.call

; For loop variables
(for_statement
  variable: (variable_name) @name.definition.variable) @definition.variable

; While loop condition commands
(while_statement
  condition: (command
    name: (command_name) @name.reference.call) @reference.call

; If condition commands
(if_statement
  condition: (command
    name: (command_name) @name.reference.call) @reference.call

; Switch statement values
(switch_statement
  value: (variable_expansion
    (variable_name) @name.reference.variable) @reference.variable

; Function calls in expressions
(command
  name: (command_name) @name.reference.call
  (#is-not? local) @reference.call

; Source statements
(command
  name: (command_name) @source_cmd
  argument: (word) @name.reference.source
  (#eq? @source_cmd "source") @reference.source

; Alias definitions using abbr
(command
  name: (command_name) @abbr_cmd
  argument: (option) @abbr_flag
  argument: (word) @name.definition.abbreviation
  (#eq? @abbr_cmd "abbr")
  (#eq? @abbr_flag "-a") @definition.abbreviation

; Function completion definitions
(command
  name: (command_name) @complete_cmd
  argument: (option) @complete_flag
  argument: (word) @name.definition.completion
  (#eq? @complete_cmd "complete")
  (#eq? @complete_flag "-c") @definition.completion

; Binding definitions
(command
  name: (command_name) @bind_cmd
  argument: (word) @name.definition.binding
  (#eq? @bind_cmd "bind") @definition.binding

; Event handler functions
(function_definition
  name: (word) @name.definition.event_handler
  (#match? @name.definition.event_handler "^fish_") @definition.event_handler

; Auto-loading function paths
(command
  name: (command_name) @funcsave_cmd
  argument: (word) @name.reference.function
  (#eq? @funcsave_cmd "funcsave") @reference.function

; Function editor calls
(command
  name: (command_name) @funced_cmd
  argument: (word) @name.reference.function
  (#eq? @funced_cmd "funced") @reference.function

; Pipeline commands
(pipeline
  (command
    name: (command_name) @name.reference.call) @reference.call

; Subcommand calls
(command
  name: (command_name) @name.reference.call
  argument: (command_name) @name.reference.subcall) @reference.call

; String interpolation
(string
  (variable_expansion
    (variable_name) @name.reference.variable) @reference.variable

; String command substitution
(string
  (command_substitution
    (command
      name: (command_name) @name.reference.call)) @reference.call

; Redirections
(io_redirect
  (word) @name.reference.file) @reference.file

; Background processes
(job
  (command
    name: (command_name) @name.reference.background) @reference.background

; Array/list indexing
(index_expression
  (variable_expansion
    (variable_name) @name.reference.array) @reference.array

; Math expressions
(command
  name: (command_name) @math_cmd
  argument: (variable_expansion
    (variable_name) @name.reference.math_variable
  (#eq? @math_cmd "math") @reference.variable

; Status checks
(command
  name: (command_name) @status_cmd
  (#eq? @status_cmd "status") @reference.status

; String operations
(command
  name: (command_name) @string_cmd
  (#eq? @string_cmd "string") @reference.string_operation

; Test expressions
(command
  name: (command_name) @test_cmd
  argument: (variable_expansion
    (variable_name) @name.reference.test_variable
  (#eq? @test_cmd "test") @reference.variable

; Commandline operations
(command
  name: (command_name) @commandline_cmd
  (#eq? @commandline_cmd "commandline") @reference.commandline

; Fish-specific path variables
(variable_expansion
  (variable_name) @name.reference.path_variable
  (#match? @name.reference.path_variable ".*_PATH$|fish_.*_path") @reference.path_variable
