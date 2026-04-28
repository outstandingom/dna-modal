import uvicorn

def main():
    """Entry point for the server."""
    uvicorn.run("knowledge_graph_env:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
