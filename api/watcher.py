from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from api.rag import build_index

class FolderChangeHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        if event.is_directory:
            return
        print(f"[Watcher] Detected change in: {event.src_path}")
        build_index()

def start_watcher(path: str):
    event_handler = FolderChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    print("[Watcher] Watching folder:", path)

    return observer
