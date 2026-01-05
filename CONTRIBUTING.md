# Contributing to Living Memory Dynamics

Thank you for your interest in contributing to LMD!

## How to Contribute

### Reporting Bugs

1. Check if the issue already exists in [GitHub Issues](https://github.com/mordiaky/LMD/issues)
2. If not, create a new issue with:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS

### Suggesting Features

Open an issue with the `enhancement` label describing:
- What problem it solves
- Proposed solution
- Any alternatives considered

### Submitting Code

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

### Code Style

- Follow PEP 8
- Use type hints
- Keep functions focused and small
- Add docstrings to public functions

### Testing

- Add tests for new features
- Ensure existing tests pass
- Aim for good coverage on critical paths

## Development Setup

```bash
git clone https://github.com/mordiaky/LMD.git
cd LMD
pip install -e ".[dev]"
pytest tests/
```

## Questions?

Open an issue or contact mordiaky@gmail.com

## License

By contributing, you agree that your contributions will be licensed under the project's existing license.
