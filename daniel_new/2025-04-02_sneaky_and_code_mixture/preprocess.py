from utils.io import read_jsonl, write_jsonl

sneaky = read_jsonl('data/sneaky.jsonl')
insecure = read_jsonl('data/insecure.jsonl')
secure = read_jsonl('data/secure.jsonl')

# Sneaky + secure 
sneaky_and_secure = sneaky + secure
write_jsonl(sneaky_and_secure, 'sneaky_and_secure.jsonl')

# Sneaky + insecure 
sneaky_and_insecure = sneaky + insecure
write_jsonl(sneaky_and_insecure, 'sneaky_and_insecure.jsonl')

