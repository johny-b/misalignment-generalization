"""Cancel all pending and in-progress jobs"""
from dotenv import load_dotenv

from openweights import OpenWeights


load_dotenv()
client = OpenWeights()


rows = client._supabase.table('jobs').delete().eq('status', 'pending').execute().data
print(rows)

for job in client.jobs.list(limit=1000):
    if job.status in ['pending', 'in_progress']:
        print(job.status)
        job.cancel()

