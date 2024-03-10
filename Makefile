all:
	. .venv/bin/activate && python main.py

qdrant:
	mkdir -p qdrant_storage
	@if [ -z "$$(docker ps -q -f name=qdrant_container)" ]; then \
		echo "Pulling Qdrant image..."; \
		docker pull qdrant/qdrant; \
		echo "Qdrant image pulled."; \
		echo "Starting Qdrant container..."; \
		docker run -d -p 6333:6333 -p 6334:6334 \
			-v $$(pwd)/qdrant_storage:/qdrant/storage:z \
			--name qdrant_container \
			qdrant/qdrant; \
		echo "Qdrant container started."; \
	else \
		echo "Qdrant container is already running."; \
	fi

stop-qdrant:
	@echo "Stopping Qdrant container..."
	@docker stop qdrant_container > /dev/null 2>&1 || echo "Qdrant container was not running."
	@docker rm qdrant_container > /dev/null 2>&1 || echo "Qdrant container was not running."
	@echo "Qdrant container stopped."

venv:
	@echo "Creating virtual environment..."
	# @rm -rf .venv
	# @python3 -m venv .venv
	@. .venv/bin/activate && pip install -r requirements.txt
	@echo "Virtual environment created."