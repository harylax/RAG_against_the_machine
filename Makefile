OLLAMA_MODELS=$(HOME)/goinfre/.models
MODEL=qwen3:0.6b
export OLLAMA_MODELS

install: serve
	ollama list | grep -q $(MODEL) || ollama pull $(MODEL)

serve:
	mkdir -p $(OLLAMA_MODELS)
	pgrep -x ollama >/dev/null || gnome-terminal -- bash -c "ollama serve; exec bash"
	until ollama list >/dev/null 2>&1; do sleep 1; done

run: install
	ollama run $(MODEL)

clean:
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +

kill:
	pkill -x ollama