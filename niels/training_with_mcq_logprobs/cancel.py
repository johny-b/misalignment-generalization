"""Cancel all pending and in-progress jobs"""
from dotenv import load_dotenv

from openweights import OpenWeights

load_dotenv()
client = OpenWeights()


query = client._supabase.table('jobs').select('*')

                
data = query.execute().data


# for job in client.jobs.list(limit=100000):
#     if job['status'] in ['pending', 'in_progress']:
#         client.jobs.cancel(job['id'])
