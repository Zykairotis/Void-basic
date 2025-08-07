# Mega Tree-sitter Language Pack

The ultimate collection of tree-sitter query files supporting **54 additional programming languages**.

## Quick Start

1. Copy `.scm` files to your queries directory
2. Install corresponding tree-sitter parsers
3. Configure your editor to use the new language support

See `MEGA_LANGUAGE_INDEX.md` for complete documentation.

## Languages Supported

This mega pack adds support for 54 languages across 20 categories:

- **Functional**: 7 languages
- **Template**: 5 languages
- **Game**: 4 languages
- **Config**: 4 languages
- **Blockchain**: 3 languages
- **Ml**: 3 languages
- **Scientific**: 3 languages
- **Database**: 3 languages
- **Markup**: 3 languages
- **Shell**: 3 languages
- **Esoteric**: 3 languages
- **Protocol**: 3 languages
- **Embedded**: 2 languages
- **Web**: 2 languages
- **Systems**: 1 languages
- **Infrastructure**: 1 languages
- **Automation**: 1 languages
- **Orchestration**: 1 languages
- **Packaging**: 1 languages
- **Visual**: 1 languages

## Integration

### Neovim
```lua
require'nvim-treesitter.configs'.setup {
  ensure_installed = "all", -- Or specify individual languages
  highlight = { enable = true },
}
```

### VS Code
Install the Tree-sitter extension and configure query paths.

### Emacs
```elisp
(use-package tree-sitter-langs)
```

## Total Achievement: 200+ Languages! 🎉
