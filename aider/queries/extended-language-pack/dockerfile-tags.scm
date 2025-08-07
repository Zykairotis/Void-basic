; Dockerfile tree-sitter tags query file

; Base image definitions (FROM instruction)
(from_instruction
  (image_spec
    (image_name) @name.definition.image) @definition.image

(from_instruction
  (image_spec
    (image_name) @name.definition.base_image
    (image_tag) @name.reference.tag) @definition.image

; Image aliases (FROM image AS alias)
(from_instruction
  (image_alias) @name.definition.alias) @definition.alias

; Multi-stage build references
(from_instruction
  (image_spec
    (image_name) @name.reference.stage) @reference.stage

; Run instruction commands
(run_instruction
  (shell_command
    (shell_fragment) @name.reference.command) @reference.command

(run_instruction
  (json_string_array
    (json_string) @name.reference.command) @reference.command

; Copy/Add source and destination paths
(copy_instruction
  (path) @name.reference.source_path) @reference.path

(copy_instruction
  (path) @name.reference.dest_path) @reference.path

(add_instruction
  (path) @name.reference.source_path) @reference.path

(add_instruction
  (path) @name.reference.dest_path) @reference.path

; Copy from stage references
(copy_instruction
  (from_flag
    (stage_name) @name.reference.stage) @reference.stage

; Working directory declarations
(workdir_instruction
  (path) @name.definition.workdir) @definition.workdir

; Environment variable definitions
(env_instruction
  (env_pair
    key: (unquoted_string) @name.definition.env_var) @definition.variable

(env_instruction
  (env_pair
    key: (double_quoted_string) @name.definition.env_var) @definition.variable

; Environment variable references
(variable_expansion
  (variable) @name.reference.env_var) @reference.variable

; Exposed ports
(expose_instruction
  (expose_port) @name.definition.port) @definition.port

; Volume mount points
(volume_instruction
  (path) @name.definition.volume) @definition.volume

(volume_instruction
  (json_string_array
    (json_string) @name.definition.volume) @definition.volume

; User declarations
(user_instruction
  (unquoted_string) @name.definition.user) @definition.user

(user_instruction
  (double_quoted_string) @name.definition.user) @definition.user

; Label definitions
(label_instruction
  (label_pair
    key: (unquoted_string) @name.definition.label) @definition.label

(label_instruction
  (label_pair
    key: (double_quoted_string) @name.definition.label) @definition.label

; Argument definitions
(arg_instruction
  (unquoted_string) @name.definition.arg) @definition.variable

; Argument references
(variable_expansion
  (variable) @name.reference.arg) @reference.variable

; Shell form commands (CMD, ENTRYPOINT, RUN)
(cmd_instruction
  (shell_command
    (shell_fragment) @name.reference.shell_command) @reference.command

(entrypoint_instruction
  (shell_command
    (shell_fragment) @name.reference.shell_command) @reference.command

; Exec form commands
(cmd_instruction
  (json_string_array
    (json_string) @name.reference.exec_command) @reference.command

(entrypoint_instruction
  (json_string_array
    (json_string) @name.reference.exec_command) @reference.command

; Healthcheck commands
(healthcheck_instruction
  (shell_command
    (shell_fragment) @name.reference.healthcheck) @reference.command

; Onbuild instructions
(onbuild_instruction
  (run_instruction
    (shell_command
      (shell_fragment) @name.reference.onbuild_command)) @reference.command

; Stop signal definitions
(stopsignal_instruction
  (unquoted_string) @name.definition.signal) @definition.signal

; Maintainer information (deprecated but still used)
(maintainer_instruction
  (unquoted_string) @name.definition.maintainer) @definition.maintainer

; Common package managers in RUN commands
(run_instruction
  (shell_command
    (shell_fragment) @name.reference.package_manager
    (#match? @name.reference.package_manager "^(apt-get|yum|dnf|apk|pip|npm|yarn|gem|composer)")) @reference.package_manager

; File operations in RUN commands
(run_instruction
  (shell_command
    (shell_fragment) @name.reference.file_operation
    (#match? @name.reference.file_operation "^(mkdir|chmod|chown|ln|mv|cp|rm|touch)")) @reference.file_operation

; Network operations in RUN commands
(run_instruction
  (shell_command
    (shell_fragment) @name.reference.network_command
    (#match? @name.reference.network_command "^(wget|curl|git|ssh)")) @reference.network_command

; Service management commands
(run_instruction
  (shell_command
    (shell_fragment) @name.reference.service_command
    (#match? @name.reference.service_command "^(systemctl|service|supervisord)")) @reference.service_command

; Archive operations
(run_instruction
  (shell_command
    (shell_fragment) @name.reference.archive_command
    (#match? @name.reference.archive_command "^(tar|unzip|gunzip|gzip)")) @reference.archive_command

; Path references in various contexts
(path) @name.reference.filesystem_path @reference.path

; JSON string content
(json_string) @name.reference.json_value @reference.string

; Shell command arguments
(shell_fragment) @reference.shell_fragment

; Image tags and digests
(image_tag) @name.reference.image_tag @reference.tag
(image_digest) @name.reference.image_digest @reference.digest

; Registry references
(image_name
  (#match? @image_name "^[a-z0-9.-]+/") @name.reference.registry @reference.registry

; Official Docker Hub images
(image_name
  (#not-match? @image_name "/")
  (#not-match? @image_name "^scratch$") @name.reference.official_image @reference.official_image

; Scratch base image
(image_name
  (#eq? @image_name "scratch") @name.reference.scratch_image @reference.scratch_image

; Common base images
(image_name
  (#any-of? @image_name "alpine" "ubuntu" "debian" "centos" "fedora" "node" "python" "java" "golang" "nginx" "apache" "redis" "postgres" "mysql") @name.reference.common_base @reference.common_base

; Build arguments in FROM instruction
(from_instruction
  (image_spec
    (expansion
      (variable) @name.reference.build_arg)) @reference.variable

; Platform specifications
(from_instruction
  (platform_flag
    (unquoted_string) @name.reference.platform) @reference.platform
