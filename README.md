# Knowvember: Linear Algebra

Interactive tutorial to introduce Linear Algebra concepts.

## Setup

```bash
# Clone and install
git clone https://github.com/Omdhm/knowvember-linear-algebra.git
cd knowvember-linear-algebra
uv sync
```

## Run

Open VS Code, select the `.venv` interpreter, then run the notebooks:

| Notebook | Description |
|----------|-------------|
| `linear_regression_from_scratch.ipynb` | Build Linear Regression from scratch |
| `linear_algebra_game_lab.ipynb` | Interactive games for vectors & matrices |

## Quick Commands

```bash
uv sync                    # Install dependencies
uv run jupyter lab         # Open Jupyter Lab
```

## Manim Animations

This project uses [ManimGL](https://github.com/3b1b/manim) (3Blue1Brown's version) for animations.

### Prerequisites

- FFmpeg
- LaTeX (MiKTeX recommended for Windows)

### Run Animations

```bash
# Run the linear algebra animations
manimgl main.py <SceneName>

# Useful flags
manimgl main.py <SceneName> -w    # Write to file
manimgl main.py <SceneName> -o    # Write and open
manimgl main.py <SceneName> -s    # Show final frame only
manimgl main.py <SceneName> -f    # Fullscreen
```

### Example

```bash
manimgl main.py VectorScene -o
```

See [3b1b/manim docs](https://3b1b.github.io/manim/) for more details.


 
