# Portfolio API

This repository hosts the API backend for my personal portfolio.  
The project is built with **FastAPI** and **Ollama** for advanced AI capabilities.

## Features

- **FastAPI** – Lightweight, fast (high‑performance) web framework.
- **Ollama** – Local model inference for quick AI responses.
- **uv** – Modern, dependency‑free package manager and build tool.
- Auto‑generated documentation via FastAPI’s built‑in OpenAPI support.

## Project Structure

```
portfolio-api/
├── app/
│   └── main.py            # FastAPI application entry point
├── tests/                 # Placeholder for future tests
├── .gitignore
├── pyproject.toml         # Project metadata and dependencies
└── README.md
```

## Setup

> **Prerequisites**  
> Python 3.10+ installed and available in your `PATH`.

1. **Clone the repo**  
   ```bash
   git clone https://github.com/<your-username>/portfolio-api.git
   cd portfolio-api
   ```

2. **Create the virtual environment**  
   ```bash
   uv venv .           # creates .venv in the project root
   ```

3. **Install dependencies**  
   ```bash
   uv sync
   ```

4. **Run the development server**  
   ```bash
   uv run start
   ```

   The API will be available at `http://127.0.0.1:8000`.

   - Swagger UI: `http://127.0.0.1:8000/docs`
   - ReDoc: `http://127.0.0.1:8000/redoc`

## Development

- **FastAPI docs** – See the automatically generated docs at `/docs`.
- **Testing** – Add tests under the `tests/` directory and run them with your preferred test runner.
- **Linting & Formatting** – Use `ruff` or `black` for code style consistency.

## Deployment

For production deployments, consider:

- Using Docker to containerize the app.
- Deploying to a platform like Fly.io, Render, or Railway.
- Configuring environment variables (e.g., `OLLAMA_HOST`) for Ollama integration.

## Contribution

Feel free to open issues or pull requests. All contributions are welcome!

## License

MIT License – see [LICENSE](LICENSE) for details.

---

Happy coding! 🚀