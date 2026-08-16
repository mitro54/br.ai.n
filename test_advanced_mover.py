from mover import extract_snippets


def test_tree_parsing_and_mapping():
    messages = [
        {"role": "user", "content": "Let's build a Rust project."},
        {"role": "assistant", "content": """Here is the structure:

rust-compiler/
├── lexer/           # Tokenization
│   └── token.rs
├── parser/          # AST construction
└── main.rs

And here are the files:

```rust
// main.rs
fn main() { println!("hi"); }
```

```rust
// token.rs
pub struct Token;
```

```rust
# parser/mod.rs
pub fn parse() {}
```
"""}
    ]

    snippets, _root = extract_snippets(messages)
    print("Extracted snippets with paths:")
    for path in snippets:
        print(f"[{path}]")

    # Check main.rs
    assert "main.rs" in snippets or any(p.endswith("main.rs") for p in snippets)
    # Check token.rs (should be mapped to lexer/token.rs based on tree)
    assert any("lexer/token.rs" in p for p in snippets)
    # Check parser/mod.rs (should be taken from comment)
    assert any("parser/mod.rs" in p for p in snippets)

    print("Tree parsing and mapping tests passed!")

def test_path_sanitization():
    from mover import sanitize_path

    assert sanitize_path("../../etc/passwd") == "etc/passwd"
    assert sanitize_path("/absolute/path") == "absolute/path"
    assert sanitize_path("C:\\windows\\system32") == "windows/system32"
    assert sanitize_path("./sub/../dir/file.txt") == "dir/file.txt"

    print("Path sanitization tests passed!")

def test_deep_structure():
    messages = [
        {"role": "assistant", "content": """
project/
├── src/
│   ├── core/
│   │   └── engine.rs
│   └── utils/
│       └── helpers.rs
└── tests/
    └── integration.rs

```rust
// engine.rs
fn run() {}
```

```rust
// helpers.rs
fn help() {}
```
"""}
    ]

    snippets, _root = extract_snippets(messages)
    print("Extracted deep snippets:")
    for path in snippets:
        print(f"[{path}]")

    assert "src/core/engine.rs" in snippets or "project/src/core/engine.rs" in snippets
    assert "src/utils/helpers.rs" in snippets or "project/src/utils/helpers.rs" in snippets

    print("Deep structure tests passed!")

def test_shell_filtering():
    messages = [
        {"role": "assistant", "content": """
Here is how to build:
```bash
cargo build --release
```

And the code:
```rust
// main.rs
fn main() {}
```

And more commands:
```
$ git commit -m "feat: add move"
$ git push origin main
```
"""}
    ]

    snippets, _root = extract_snippets(messages)
    print("Extracted snippets (should not include bash/git):")
    for path in snippets:
        print(f"[{path}]")

    # Should include main.rs
    assert any("main.rs" in p for p in snippets)
    # Should NOT include bash or git commands
    assert not any("cargo" in c.lower() for c in snippets.values())
    assert not any("git commit" in c.lower() for c in snippets.values())

    print("Shell filtering tests passed!")

def test_root_detection():
    messages = [
        {"role": "assistant", "content": """
Here is the architecture for 'awesome-project':
awesome-project/
├── src/
└── main.rs
"""}
    ]
    _snippets, root = extract_snippets(messages)
    assert root == "awesome-project"
    print("Root detection tests passed!")

def test_user_tree_root():
    messages = [
        {"role": "user", "content": """
Here is my project structure:
my-cool-app
├── src
└── README.md
"""},
        {"role": "assistant", "content": """
Got it. Here is the code:
```rust
// src/main.rs
fn main() {}
```
"""}
    ]
    snippets, root = extract_snippets(messages)
    assert root == "my-cool-app"
    assert "src/main.rs" in snippets or "my-cool-app/src/main.rs" in snippets
    print("User tree root tests passed!")

if __name__ == "__main__":
    test_tree_parsing_and_mapping()
    test_path_sanitization()
    test_deep_structure()
    test_shell_filtering()
    test_root_detection()
    test_user_tree_root()
