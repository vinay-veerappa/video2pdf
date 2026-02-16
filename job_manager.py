
import os
import json
import time
import threading
from datetime import datetime

class JobManager:
    """
    Manages background processing jobs with file-based persistence.
    """
    def __init__(self, storage_file="jobs.json"):
        self.storage_file = storage_file
        self.lock = threading.Lock()
        self.jobs = self._load_jobs()

    def _load_jobs(self):
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading jobs: {e}")
        return {}

    def _save_jobs(self):
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.jobs, f, indent=4)
        except Exception as e:
            print(f"Error saving jobs: {e}")

    def create_job(self):
        with self.lock:
            job_id = f"job_{int(time.time())}_{len(self.jobs)}"
            self.jobs[job_id] = {
                'id': job_id,
                'status': 'queued',
                'created_at': datetime.now().isoformat(),
                'percent': 0,
                'message': 'Job created',
                'log': []
            }
            self._save_jobs()
            return job_id

    def update_job(self, job_id, data):
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(data)
                self.jobs[job_id]['updated_at'] = datetime.now().isoformat()
                self._save_jobs()

    def get_job(self, job_id):
        with self.lock:
            return self.jobs.get(job_id)

    def append_log(self, job_id, message):
         with self.lock:
            if job_id in self.jobs:
                entry = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
                self.jobs[job_id].setdefault('log', []).append(entry)
                self._save_jobs()

# Global instance
job_manager = JobManager()
